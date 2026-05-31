"""
agente_lib.py — peças reutilizáveis do módulo Agente.

Stack:
    - LangChain (@tool, ChatOllama com .bind_tools)
    - LangGraph (StateGraph, ToolNode, MemorySaver)
    - LangSmith (tracing automático via .env)

Camadas:
    1. Memória (curto prazo via state do grafo; longo prazo em JSON)
    2. Guardrails (entrada + saída) — nodes do grafo
    3. Observabilidade (LangSmith + JSONL local)
    4. Ferramentas (@tool do LangChain)

O grafo:

    START → guardrail_in ─[ok]──► agente ─[chamou tool?]──► tools ──┐
              │                     │                                │
              │[bloqueado]          │[respondeu]                     │
              ▼                     ▼                                │
             END               guardrail_out → END ◄─────────────────┘
"""

from __future__ import annotations

import ast
import importlib.util
import json
import operator as op
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Callable, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ---------------------------------------------------------------------------
# Paths e .env
# ---------------------------------------------------------------------------
DIR_BASE = Path(__file__).resolve().parent
RAIZ = DIR_BASE.parent
load_dotenv(RAIZ / ".env")  # ativa LangSmith se configurado

DIR_LOGS = DIR_BASE / "logs"
DIR_LOGS.mkdir(exist_ok=True)
ARQUIVO_LOG = DIR_LOGS / "agente.jsonl"

ARQUIVO_MEMORIA_LP = DIR_BASE / "memoria_longo_prazo.json"

MODELO_LLM = "llama3.2"

# Import dinâmico de rag_lib do módulo 2 (a pasta tem espaço no nome)
CAMINHO_RAG_LIB = RAIZ / "2. RAG" / "rag_lib.py"


def _carregar_rag_lib():
    if not CAMINHO_RAG_LIB.exists():
        return None
    spec = importlib.util.spec_from_file_location("rag_lib", CAMINHO_RAG_LIB)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rag_lib"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


rag_lib = _carregar_rag_lib()


# ===========================================================================
# 1. Memória de longo prazo (JSON em disco)
# ===========================================================================
class MemoriaLongoPrazo:
    """Fatos persistidos em JSON entre sessões."""

    def __init__(self, caminho: Path = ARQUIVO_MEMORIA_LP):
        self.caminho = caminho
        self._dados: dict[str, str] = self._carregar()

    def _carregar(self) -> dict[str, str]:
        if not self.caminho.exists():
            return {}
        try:
            return json.loads(self.caminho.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _salvar(self) -> None:
        self.caminho.write_text(
            json.dumps(self._dados, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def lembrar(self, chave: str, valor: str) -> None:
        self._dados[chave] = valor
        self._salvar()

    def buscar(self, chave: str) -> str | None:
        return self._dados.get(chave)

    def tudo(self) -> dict[str, str]:
        return dict(self._dados)

    def esquecer(self, chave: str) -> bool:
        if chave in self._dados:
            del self._dados[chave]
            self._salvar()
            return True
        return False


memoria_lp_global = MemoriaLongoPrazo()


# ===========================================================================
# 2. Guardrails (puro Python, sem LLM)
# ===========================================================================
LIMITE_CARS_ENTRADA = 4000

PADROES_INJECTION = [
    r"ignore (todas as )?(instruç(ões|oes)|prompts) anteriores",
    r"esqueça (a|sua) persona",
    r"você agora é",
    r"voce agora e",
    r"ignore the (previous|above) (instructions|prompt)",
    r"act as a (different|new)",
    r"system prompt",
]

GATILHOS_PROIBIDOS = [
    "como fabricar bomba",
    "como invadir",
    "número de cartão",
]

REGEX_CPF = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
REGEX_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def validar_entrada(texto: str) -> tuple[bool, str | None]:
    """Retorna (permitido, motivo_se_bloqueado)."""
    texto = (texto or "").strip()
    if not texto:
        return False, "entrada vazia"
    if len(texto) > LIMITE_CARS_ENTRADA:
        return False, "entrada acima do limite"
    baixo = texto.lower()
    for padrao in PADROES_INJECTION:
        if re.search(padrao, baixo):
            return False, f"possível prompt injection ({padrao!r})"
    for gatilho in GATILHOS_PROIBIDOS:
        if gatilho in baixo:
            return False, f"conteúdo proibido na entrada: {gatilho!r}"
    return True, None


def _mascarar_pii(texto: str) -> str:
    texto = REGEX_CPF.sub("[CPF_MASCARADO]", texto)
    texto = REGEX_EMAIL.sub("[EMAIL_MASCARADO]", texto)
    return texto


def sanitizar_saida(texto: str) -> tuple[str, str | None]:
    """Retorna (texto_final, motivo_se_substituido)."""
    baixo = texto.lower()
    for gatilho in GATILHOS_PROIBIDOS:
        if gatilho in baixo:
            return "[RESPOSTA BLOQUEADA pelo guardrail]", f"saída continha {gatilho!r}"
    return _mascarar_pii(texto), None


# ===========================================================================
# 3. Observabilidade local (complementa o LangSmith)
# ===========================================================================
class LoggerJSONL:
    def __init__(self, arquivo: Path = ARQUIVO_LOG):
        self.arquivo = arquivo
        self.sessao_id = uuid.uuid4().hex[:12]

    def evento(self, tipo: str, **dados: Any) -> None:
        registro = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "sessao_id": self.sessao_id,
            "tipo": tipo,
            **dados,
        }
        with self.arquivo.open("a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False, default=str) + "\n")


logger_jsonl = LoggerJSONL()


# ===========================================================================
# 4. Ferramentas (@tool do LangChain)
# ===========================================================================
@tool
def rag_buscar(pergunta: str, top_k: int = 3) -> str:
    """Busca trechos relevantes nos PDFs indexados pelo módulo RAG.

    Use sempre que a pergunta exigir conhecimento dos documentos da empresa.
    """
    if rag_lib is None:
        return "Erro: rag_lib não encontrado. Rode o módulo 2 (RAG) antes."
    resultados = rag_lib.recuperar(pergunta, k=top_k)
    if not resultados:
        return "Base RAG vazia. Rode '2. RAG/main.py' para indexar."
    linhas = []
    for doc, score in resultados:
        fonte = Path(doc.metadata.get("source", "?")).name
        pagina = doc.metadata.get("page", "?")
        linhas.append(
            f"[{fonte} p.{pagina} (dist={score:.3f})]\n{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(linhas)


@tool
def lembrar_fato(chave: str, valor: str) -> str:
    """Salva um fato sobre o usuário na memória de longo prazo (chave/valor)."""
    memoria_lp_global.lembrar(chave, valor)
    return f"OK, lembrei: {chave} = {valor}"


@tool
def hora_atual() -> str:
    """Retorna a hora atual em UTC (ISO 8601)."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


# Calculadora segura via AST
_OPERADORES_SEGUROS: dict[type, Callable[..., Any]] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _avaliar_expr(no: ast.AST) -> float:
    if isinstance(no, ast.Constant) and isinstance(no.value, (int, float)):
        return no.value
    if isinstance(no, ast.BinOp) and type(no.op) in _OPERADORES_SEGUROS:
        return _OPERADORES_SEGUROS[type(no.op)](_avaliar_expr(no.left), _avaliar_expr(no.right))
    if isinstance(no, ast.UnaryOp) and type(no.op) in _OPERADORES_SEGUROS:
        return _OPERADORES_SEGUROS[type(no.op)](_avaliar_expr(no.operand))
    raise ValueError("expressão não suportada")


@tool
def calcular(expressao: str) -> str:
    """Avalia uma expressão matemática simples (aritmética: + - * / ** %)."""
    try:
        arvore = ast.parse(expressao, mode="eval")
        return str(_avaliar_expr(arvore.body))
    except Exception as e:
        return f"Erro ao calcular '{expressao}': {e}"


FERRAMENTAS = [rag_buscar, lembrar_fato, hora_atual, calcular]


# ===========================================================================
# 5. LangGraph — estado e nodes
# ===========================================================================
class EstadoAgente(TypedDict):
    """O estado que circula pelo grafo."""

    messages: Annotated[list[BaseMessage], add_messages]
    entrada_bloqueada: bool
    motivo_bloqueio: str | None


SYSTEM_PROMPT = """Você é um assistente útil e factual.

Quando a pergunta exigir conhecimento dos documentos da empresa, use OBRIGATORIAMENTE a ferramenta `rag_buscar` antes de responder. Cite a fonte (arquivo e página) no final.

Quando o usuário compartilhar um fato pessoal duradouro (preferência, nome, contexto profissional), use `lembrar_fato` para gravar.

Se não souber a resposta mesmo após usar as ferramentas, diga claramente que não sabe.

FATOS QUE VOCÊ JÁ SABE SOBRE O USUÁRIO (memória de longo prazo):
{fatos}
"""


def montar_system_message() -> SystemMessage:
    fatos = memoria_lp_global.tudo()
    fatos_str = "\n".join(f"- {k}: {v}" for k, v in fatos.items()) or "(nenhum)"
    return SystemMessage(content=SYSTEM_PROMPT.format(fatos=fatos_str))


def _llm_com_tools() -> ChatOllama:
    """LLM com as ferramentas do agente já registradas."""
    llm = ChatOllama(model=MODELO_LLM, temperature=0.1)
    return llm.bind_tools(FERRAMENTAS)


# ---------- Nodes ----------
def node_guardrail_in(estado: EstadoAgente) -> dict:
    """Roda a validação de entrada na última HumanMessage."""
    ultima_humana = next(
        (m for m in reversed(estado["messages"]) if isinstance(m, HumanMessage)), None
    )
    if ultima_humana is None:
        return {"entrada_bloqueada": False, "motivo_bloqueio": None}

    permitido, motivo = validar_entrada(ultima_humana.content)
    logger_jsonl.evento(
        "guardrail_entrada",
        permitido=permitido,
        motivo=motivo,
        tamanho=len(ultima_humana.content),
    )
    if not permitido:
        return {
            "entrada_bloqueada": True,
            "motivo_bloqueio": motivo,
            "messages": [AIMessage(content=f"Não posso processar essa entrada: {motivo}")],
        }
    return {"entrada_bloqueada": False, "motivo_bloqueio": None}


def node_agente(estado: EstadoAgente) -> dict:
    """Chama o LLM com tools."""
    msgs = [montar_system_message()] + estado["messages"]
    resposta = _llm_com_tools().invoke(msgs)
    logger_jsonl.evento(
        "chamada_llm",
        tem_tool_call=bool(getattr(resposta, "tool_calls", None)),
        tools=[t["name"] for t in getattr(resposta, "tool_calls", []) or []],
    )
    return {"messages": [resposta]}


def node_guardrail_out(estado: EstadoAgente) -> dict:
    """Sanitiza a última AIMessage (mascara PII, bloqueia conteúdo proibido)."""
    ultima = estado["messages"][-1]
    if not isinstance(ultima, AIMessage) or not isinstance(ultima.content, str):
        return {}

    texto_final, motivo = sanitizar_saida(ultima.content)
    logger_jsonl.evento(
        "guardrail_saida",
        modificou=texto_final != ultima.content,
        motivo=motivo,
    )
    if texto_final == ultima.content:
        return {}
    # Substituímos a última mensagem mantendo o mesmo id
    return {"messages": [AIMessage(content=texto_final, id=ultima.id)]}


# ---------- Edges condicionais ----------
def roteador_pos_guardrail_in(estado: EstadoAgente) -> str:
    return "fim" if estado.get("entrada_bloqueada") else "agente"


def roteador_pos_agente(estado: EstadoAgente) -> str:
    ultima = estado["messages"][-1]
    if isinstance(ultima, AIMessage) and getattr(ultima, "tool_calls", None):
        return "tools"
    return "guardrail_out"


# ---------- Compilação do grafo ----------
def construir_grafo():
    grafo = StateGraph(EstadoAgente)

    grafo.add_node("guardrail_in", node_guardrail_in)
    grafo.add_node("agente", node_agente)
    grafo.add_node("tools", ToolNode(FERRAMENTAS))
    grafo.add_node("guardrail_out", node_guardrail_out)

    grafo.add_edge(START, "guardrail_in")
    grafo.add_conditional_edges(
        "guardrail_in",
        roteador_pos_guardrail_in,
        {"agente": "agente", "fim": END},
    )
    grafo.add_conditional_edges(
        "agente",
        roteador_pos_agente,
        {"tools": "tools", "guardrail_out": "guardrail_out"},
    )
    grafo.add_edge("tools", "agente")
    grafo.add_edge("guardrail_out", END)

    # MemorySaver = checkpoint em memória. Para persistência, troque por
    # SqliteSaver / PostgresSaver.
    return grafo.compile(checkpointer=MemorySaver())
