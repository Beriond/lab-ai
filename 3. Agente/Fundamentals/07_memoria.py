"""
Memória de CURTO e LONGO prazo — e o caso SEM memória (chat interativo)

Três modos para comparar:

0) SEM memória
   - Nenhum checkpointer, nenhum thread_id, nenhum JSON.
   - Cada turno é uma chamada independente ao LLM.
   - O agente não lembra NADA do que foi dito antes — nem no mesmo
     chat. Útil para ver o contraste com os modos abaixo.

1) Memória de CURTO prazo (sessão)
   - Histórico de mensagens da conversa atual.
   - Vive em RAM via `InMemorySaver`, indexado por `thread_id`.
   - Some quando o programa fecha.
   - O `thread_id` é o "id da conversa": mesmo id = mesmo histórico.

2) Memória de LONGO prazo (entre sessões)
   - FATOS que o agente precisa lembrar SEMPRE.
   - Persistida num JSON em disco (`memoria_longa.json`).
   - A cada turno rodamos um EXTRATOR com saída estruturada que
     identifica TODOS os fatos pessoais da mensagem (nome, idade,
     profissão, hobby...). É mais confiável do que dar uma tool ao
     agente — modelos pequenos (llama3.2) costumam pegar uns
     fatos e esquecer outros.
   - O system prompt do agente injeta os fatos atuais a cada turno —
     assim o modelo já "sabe" das informações antes de responder.

Como rodar:
    python 07_memoria.py sem      # chat sem nenhuma memória
    python 07_memoria.py curta    # chat com memória de curto prazo
    python 07_memoria.py longa    # chat com memória de longo prazo

Digite /sair para encerrar o chat.
Rode o modo `longa` duas vezes para ver os fatos sobreviverem ao
fechamento do programa.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict
from pydantic import BaseModel, Field

# Carrega .env da raiz do projeto para ativar o tracing do LangSmith
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

MODELO = "llama3.2"
ARQUIVO_MEMORIA = Path(__file__).parent / "memoria_longa.json"


# ---------------------------------------------------------------------------
# Persistência da memória longa
# ---------------------------------------------------------------------------
def carregar_memoria() -> dict:
    if ARQUIVO_MEMORIA.exists():
        return json.loads(ARQUIVO_MEMORIA.read_text(encoding="utf-8"))
    return {}


def salvar_memoria(dados: dict) -> None:
    ARQUIVO_MEMORIA.write_text(
        json.dumps(dados, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Extrator de fatos via SAÍDA ESTRUTURADA.
#
# Por que não usar uma tool aqui?
#   Modelos pequenos (llama3.2) frequentemente "esquecem" de chamar a tool
#   ou só extraem parte dos fatos (ex.: pega `profissao` mas ignora
#   `idade`). Saída estruturada é determinística: o modelo é OBRIGADO a
#   preencher o schema, então sempre devolve um dicionário — vazio se
#   não houver fatos.
# ---------------------------------------------------------------------------
class FatosUsuario(BaseModel):
    """Fatos pessoais extraídos da mensagem do usuário."""

    fatos: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Dicionário com TODOS os fatos pessoais mencionados na mensagem. "
            "Chaves em snake_case, sem acentos, em português. "
            "Exemplos de chaves: nome, idade, profissao, cidade, estado, "
            "estado_civil, filhos, hobby, bebida_preferida, comida_preferida, "
            "alergia, time_futebol, religiao, signo, telefone, email, "
            "animal_estimacao, aniversario. "
            "Valores em texto curto. Se a mensagem NÃO contém nenhum fato "
            "pessoal sobre o usuário, devolva um dicionário VAZIO {}."
        ),
    )


def extrair_fatos(texto: str) -> Dict[str, str]:
    """Roda um LLM com saída estruturada para identificar TODOS os fatos
    pessoais na mensagem. Retorna um dicionário (possivelmente vazio)."""
    llm = ChatOllama(model=MODELO, temperature=0)
    prompt = (
        "Sua tarefa: extrair fatos pessoais da mensagem do usuário.\n"
        "Inclua TUDO que for informação sobre ELE: nome, idade, profissão, "
        "cidade, hobbies, preferências, família, datas, etc.\n"
        "NÃO invente fatos que não estão na mensagem.\n"
        "Se a mensagem for apenas uma pergunta, devolva {}.\n\n"
        f'Mensagem: "{texto}"'
    )
    # Tag separada do agente para conseguir filtrar só o extrator no LangSmith
    config = {
        "tags": ["aula:07", "demo:memoria:extrator"],
        "metadata": {"aula": "07", "modo": "extrator"},
    }
    resultado = llm.with_structured_output(FatosUsuario).invoke(prompt, config=config)
    return resultado.fatos


def gravar_fatos(novos: Dict[str, str]) -> None:
    """Mescla os novos fatos no JSON em disco."""
    dados = carregar_memoria()
    dados.update(novos)
    salvar_memoria(dados)


# ---------------------------------------------------------------------------
# Construção dos agentes
# ---------------------------------------------------------------------------
def criar_agente_sem_memoria():
    """Agente SEM checkpointer e sem tools. Cada invoke é independente:
    o agente NÃO lembra do turno anterior nem no mesmo chat."""
    llm = ChatOllama(model=MODELO, temperature=0)
    return create_agent(
        model=llm,
        tools=[],
        system_prompt="Você é um assistente simpático. Responda em português.",
    )


def criar_agente_curto():
    """Agente sem tools, com checkpointer em RAM por thread_id."""
    llm = ChatOllama(model=MODELO, temperature=0)
    return create_agent(
        model=llm,
        tools=[],
        system_prompt="Você é um assistente simpático. Responda em português.",
        checkpointer=InMemorySaver(),
    )


def criar_agente_longo():
    """Agente sem tools. Os fatos atuais do JSON já entram no system
    prompt, então reconstruímos o agente a cada turno para refletir
    fatos que acabaram de ser gravados pelo extrator."""
    fatos = carregar_memoria()
    if fatos:
        bloco_fatos = "\n".join(f"- {k}: {v}" for k, v in fatos.items())
    else:
        bloco_fatos = "(nenhum fato salvo ainda)"

    prompt = (
        "Você é um assistente pessoal com memória.\n"
        "Fatos que você já sabe sobre o usuário:\n"
        f"{bloco_fatos}\n\n"
        "Quando o usuário fizer uma pergunta sobre ele, RESPONDA usando os "
        "fatos da lista acima — não invente. Se um fato não está na lista, "
        "diga que ainda não sabe."
    )

    llm = ChatOllama(model=MODELO, temperature=0)
    return create_agent(model=llm, tools=[], system_prompt=prompt)


# ---------------------------------------------------------------------------
# Funções de conversa
# ---------------------------------------------------------------------------
def conversar_sem_memoria(agente, texto: str) -> str:
    """Invoca o agente sem nenhuma config: cada chamada é independente."""
    config = {
        "tags": ["aula:07", "demo:memoria:sem"],
        "metadata": {"aula": "07", "modo": "sem"},
    }
    saida = agente.invoke({"messages": [HumanMessage(content=texto)]}, config)
    return saida["messages"][-1].content


def conversar_curto(agente, thread_id: str, texto: str) -> str:
    """Invoca o agente preservando o histórico via thread_id.
    No LangSmith, runs com o mesmo thread_id ficam agrupados em uma Thread."""
    config = {
        "configurable": {"thread_id": thread_id},
        "tags": ["aula:07", "demo:memoria:curta"],
        "metadata": {"aula": "07", "modo": "curta", "thread_id": thread_id},
    }
    saida = agente.invoke({"messages": [HumanMessage(content=texto)]}, config)
    return saida["messages"][-1].content


def conversar_longo(agente, texto: str) -> str:
    """Invoca o agente — os fatos já estão no system prompt."""
    config = {
        "tags": ["aula:07", "demo:memoria:longa"],
        "metadata": {"aula": "07", "modo": "longa"},
    }
    saida = agente.invoke({"messages": [HumanMessage(content=texto)]}, config)
    return saida["messages"][-1].content


# ---------------------------------------------------------------------------
# Demos interativas
# ---------------------------------------------------------------------------
def demo_sem_memoria() -> None:
    print("\n" + "=" * 60)
    print("  DEMO 0 — SEM memória (cada turno é independente)")
    print("=" * 60)
    print("O agente NÃO tem checkpointer. Cada invoke começa do zero.")
    print("Diga seu nome, depois pergunte qual é o seu nome — ele não saberá.")
    print("Digite /sair para encerrar.\n")

    agente = criar_agente_sem_memoria()
    while True:
        entrada = input("Você: ").strip()
        if not entrada or entrada == "/sair":
            break
        resposta = conversar_sem_memoria(agente, entrada)
        print(f"Bot: {resposta}\n")


def demo_memoria_curta() -> None:
    print("\n" + "=" * 60)
    print("  DEMO 1 — Memória de CURTO prazo (por thread_id, em RAM)")
    print("=" * 60)
    print("Vamos abrir DUAS conversas. Na primeira, fale algo sobre você")
    print("(seu nome, gosto, etc.). Na segunda — thread nova — pergunte")
    print("a mesma coisa: o agente terá esquecido.")
    print("Digite /sair para encerrar cada chat.\n")

    agente = criar_agente_curto()

    print("--- CHAT A (thread_id=chat-a) ---")
    while True:
        entrada = input("Você: ").strip()
        if not entrada or entrada == "/sair":
            break
        resposta = conversar_curto(agente, "chat-a", entrada)
        print(f"Bot: {resposta}\n")

    print("\n--- CHAT B (thread_id=chat-b — thread NOVA) ---")
    print("Faça uma pergunta que dependia do chat anterior.\n")
    while True:
        entrada = input("Você: ").strip()
        if not entrada or entrada == "/sair":
            break
        resposta = conversar_curto(agente, "chat-b", entrada)
        print(f"Bot: {resposta}\n")


def demo_memoria_longa() -> None:
    print("\n" + "=" * 60)
    print("  DEMO 2 — Memória de LONGO prazo (JSON em disco)")
    print("=" * 60)
    print(f"Arquivo: {ARQUIVO_MEMORIA.name}")
    print(f"Memória atual: {carregar_memoria()}")
    print("\nConte coisas sobre você (nome, idade, profissão, hobbies...).")
    print("A cada turno rodamos DOIS passos:")
    print("  1) extrator (saída estruturada) → grava fatos no JSON")
    print("  2) agente → responde usando os fatos atuais do JSON")
    print("Depois pergunte algo sobre você — ele responde usando o JSON.")
    print("Digite /sair para encerrar.\n")

    while True:
        entrada = input("Você: ").strip()
        if not entrada or entrada == "/sair":
            break

        # 1) Extrai fatos ANTES de invocar o agente. Como o extrator usa
        #    saída estruturada, ele é confiável (não esquece campos).
        novos = extrair_fatos(entrada)
        if novos:
            gravar_fatos(novos)
            print(f"[EXTRATOR] +{novos}")
        else:
            print("[EXTRATOR] nenhum fato novo")

        # 2) Reconstrói o agente para o system prompt refletir os fatos
        #    recém-salvos, depois responde.
        agente = criar_agente_longo()
        resposta = conversar_longo(agente, entrada)
        print(f"\nBot: {resposta}\n")

    print(f"\nMemória final em disco: {carregar_memoria()}")
    print("Dica: rode o script de novo — os fatos continuam lá.")


# ---------------------------------------------------------------------------
# Entrada via linha de comando
# ---------------------------------------------------------------------------
MODOS = {
    "sem":   demo_sem_memoria,
    "curta": demo_memoria_curta,
    "longa": demo_memoria_longa,
}


def uso() -> None:
    print("Uso: python 07_memoria.py <modo>")
    print("Modos disponíveis:")
    print("  sem    — chat sem nenhuma memória (cada turno isolado)")
    print("  curta — chat com memória de curto prazo (sessão, em RAM)")
    print("  longa — chat com memória de longo prazo (JSON em disco)")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in MODOS:
        uso()
        sys.exit(1)
    MODOS[sys.argv[1]]()
