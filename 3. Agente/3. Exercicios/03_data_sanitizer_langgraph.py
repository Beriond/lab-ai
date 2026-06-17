"""
Exercício 03 — Sanitização de base de dados com LangGraph

Objetivo:
    Mostrar que LangGraph NÃO é só para "agentes com LLM". O grafo de
    estado também é uma forma elegante de modelar PIPELINES de dados,
    com:
        - estado tipado (TypedDict)
        - nós isolados e testáveis
        - logs por nó
        - rastreabilidade no LangSmith (se .env estiver configurado)

Pipeline a construir:

    START
      │
      ▼
    [ler_csv]          lê `cadastro.csv` (ISO-8859-1, separador `;`)
      │
      ▼
    [tratar_nomes]     cria `nome_tratado` chamando um LLM por linha:
      │                LLM aplica as regras (sem acento, trim,
      │                espaços internos viram `_`, minúsculo)
      ▼
    [validar_emails]   cria `email_valido` ("sim"/"nao") chamando um LLM
      │                por linha (contém '@' E termina em '.com'/'.com.br')
      ▼
    [salvar_csv]       grava `cadastro_nova.csv` no MESMO diretório
      │
      ▼
     END

Regras do enunciado:
    1. Ler `cadastro.csv` (encoding ISO-8859-1 / latin-1, separador `;`).
    2. Criar coluna `nome_tratado` aplicando as regras de SYSTEM_TRATAR_NOME.
    3. Criar coluna `email_valido` com "sim"/"nao" segundo SYSTEM_VALIDAR_EMAIL.
    4. Salvar como `cadastro_nova.csv` no mesmo diretório (UTF-8, `;`).

Documentação de referência:
    - LangGraph (StateGraph):
        https://langchain-ai.github.io/langgraph/
    - csv (DictReader / DictWriter):
        https://docs.python.org/3/library/csv.html
    - LangChain Ollama (ChatOllama):
        https://python.langchain.com/docs/integrations/chat/ollama/

Estrutura do script:
    1) Imports (csv, pathlib, typing, langgraph, langchain_ollama, ...).
    2) Constante MODELO e instância `llm`.
    3) Definir a TypedDict `EstadoSanitizacao` com:
         - caminho_in, caminho_out (str)
         - linhas (List[dict])
         - total (int)
         - logs (Annotated[List[str], operator.add])   <- importante!
    4) System prompts (já prontos abaixo) e duas funções wrapper que
       conversam com o LLM:
         - tratar_nome_llm(nome) -> str
         - email_e_valido_llm(email) -> bool
    5) Implementar os 4 NÓS — cada um recebe `state` e devolve um dict
       PARCIAL com as atualizações:
         - ler_csv(state)
         - tratar_nomes(state)
         - validar_emails(state)
         - salvar_csv(state)
    6) Função `construir_grafo()` que monta o StateGraph
       (START -> ler_csv -> tratar_nomes -> validar_emails -> salvar_csv -> END).
    7) Função `main()` que invoca o grafo e imprime um resumo + preview.

Como rodar:
    python "3. Agente/3. Exercicios/03_data_sanitizer_langgraph_aluno.py"
"""
from __future__ import annotations
import csv
import operator
from pathlib import Path
from typing import Annotated, List, TypedDict

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph


MODELO = "llama3.2"
llm = ChatOllama(model=MODELO, temperature=0)


# ---------------------------------------------------------------------------
# 1. ESTADO COMPARTILHADO ENTRE OS NÓS
#
#    Dica: `Annotated[List[str], operator.add]` faz o LangGraph CONCATENAR
#    listas vindas de nós diferentes em vez de sobrescrever — perfeito
#    para acumular logs ao longo do pipeline.
#
#    Campos esperados:
#       - caminho_in: str
#       - caminho_out: str
#       - linhas: List[dict]
#       - total: int
#       - logs: Annotated[List[str], operator.add]
# ---------------------------------------------------------------------------
class EstadoSanitizacao(TypedDict):
    ## aqui coloque seu código ###
    pass  # remova este `pass` ao declarar os campos acima


# ---------------------------------------------------------------------------
# 2. PROMPTS DO LLM
#    Os system prompts abaixo já estão prontos — eles documentam as regras
#    de negócio. Você precisa implementar as duas funções wrapper que
#    chamam o LLM com esses prompts.
# ---------------------------------------------------------------------------

# Escreva o prompt adequado para realizar a tarefa
SYSTEM_TRATAR_NOME = (
    "Você é um normalizador de nomes próprios.\n"
)

# Escreva o prompt adequado para realizar a tarefa
SYSTEM_VALIDAR_EMAIL = (
    "Você é um validador de e-mails.\n"
)


def tratar_nome_llm(nome: str) -> str:
    """Pede ao LLM para normalizar o nome (sem acento, trim, `_`, minúsculo).

    Dica:
        - Se `nome` vier vazio, devolver "" sem chamar o LLM.
        - Use llm.invoke([SystemMessage(...), HumanMessage(...)]).
        - Faça uma limpeza defensiva da resposta (strip, remover aspas,
          forçar minúsculo) antes de devolver.
    """
    ## aqui coloque seu código ###


def email_e_valido_llm(email: str) -> bool:
    """Pede ao LLM para classificar o e-mail como válido ('sim') ou não.

    Dica:
        - Se `email` vier vazio, devolver False sem chamar o LLM.
        - Pegue a primeira palavra alfabética da resposta para evitar
          que pontuação ou explicações extras quebrem o parser.
        - Devolver True se a primeira palavra for "sim", False caso contrário.
    """
    ## aqui coloque seu código ###


# ---------------------------------------------------------------------------
# 3. NÓS DO GRAFO
#
#    Convenção: cada nó recebe `state: EstadoSanitizacao` e devolve um
#    dict PARCIAL com as atualizações que quer aplicar no estado.
#    Tudo que não for retornado, permanece igual.
# ---------------------------------------------------------------------------
def ler_csv(state: EstadoSanitizacao) -> dict:
    """Lê o CSV de entrada em modo dict (cada linha = um dicionário).

    O arquivo está em ISO-8859-1 com separador `;` — tratamos isso aqui
    para que o resto do grafo receba dados "limpos".
    """
    caminho = Path(state["caminho_in"])
    print(f"\n[1/4] Lendo CSV: {caminho.name}")

    with caminho.open(encoding="latin-1", newline="") as f:
        leitor = csv.DictReader(f, delimiter=";")
        linhas = [dict(linha) for linha in leitor]

    total = len(linhas)
    print(f"      {total} linhas lidas, colunas: {list(linhas[0].keys()) if linhas else []}")

    return {
        "linhas": linhas,
        "total": total,
        "logs": [f"[ler_csv] {total} linhas lidas de {caminho.name}"],
    }


def tratar_nomes(state: EstadoSanitizacao) -> dict:
    """Adiciona a coluna `nome_tratado` em cada linha — via LLM.

    Cada nome é enviado individualmente ao LLM, que aplica as regras
    descritas em `SYSTEM_TRATAR_NOME` (sem acento, trim, minúsculo).
    """
    print("\n[2/4] Tratando coluna `nome` -> `nome_tratado` (via LLM)")

    linhas = state["linhas"]
    for i, linha in enumerate(linhas, start=1):
        nome_original = linha.get("nome", "")
        linha["nome_tratado"] = tratar_nome_llm(nome_original)
        print(f"      ({i}/{len(linhas)}) {nome_original!r} -> {linha['nome_tratado']!r}")

    return {
        "linhas": linhas,
        "logs": [f"[tratar_nomes] {len(linhas)} nomes normalizados via LLM"],
    }


def validar_emails(state: EstadoSanitizacao) -> dict:
    """Adiciona a coluna `email_valido` ('sim'/'nao') — via LLM.

    Cada e-mail é enviado ao LLM, que decide se é válido segundo as regras
    descritas em `SYSTEM_VALIDAR_EMAIL` (contém '@' e termina em '.com'
    ou '.com.br').
    """
    print("\n[3/4] Validando coluna `e-mail` -> `email_valido` (via LLM)")

    linhas = state["linhas"]
    validos = 0
    for i, linha in enumerate(linhas, start=1):
        # A coluna no CSV se chama "e-mail" (com hífen!) — atenção.
        email = linha.get("e-mail", "")
        ok = email_e_valido_llm(email)
        linha["email_valido"] = "sim" if ok else "nao"
        if ok:
            validos += 1
        print(f"      ({i}/{len(linhas)}) {email!r} -> {linha['email_valido']}")

    invalidos = len(linhas) - validos
    print(f"      válidos: {validos}, inválidos: {invalidos}")

    return {
        "linhas": linhas,
        "logs": [f"[validar_emails] válidos={validos}, inválidos={invalidos} (via LLM)"],
    }


def salvar_csv(state: EstadoSanitizacao) -> dict:
    """Grava o resultado em `cadastro_nova.csv` no mesmo diretório."""
    caminho = Path(state["caminho_out"])
    print(f"\n[4/4] Salvando: {caminho.name}")

    linhas = state["linhas"]
    if not linhas:
        return {"logs": ["[salvar_csv] nenhuma linha — nada a gravar"]}

    # Preserva a ordem original das colunas e acrescenta as novas no fim.
    colunas_originais = [c for c in linhas[0].keys() if c not in ("nome_tratado", "email_valido")]
    colunas = colunas_originais + ["nome_tratado", "email_valido"]

    # Salvamos em UTF-8 (padrão moderno) com separador `;` para manter
    # compatibilidade com o Excel brasileiro.
    with caminho.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=colunas, delimiter=";")
        writer.writeheader()
        writer.writerows(linhas)

    print(f"      OK — {len(linhas)} linhas gravadas em {caminho}")
    return {"logs": [f"[salvar_csv] {len(linhas)} linhas em {caminho.name}"]}

# ---------------------------------------------------------------------------
# 4. MONTAGEM DO GRAFO
#
#    Construa um StateGraph(EstadoSanitizacao), adicione os 4 nós e
#    ligue as arestas em sequência:
#       START -> ler_csv -> tratar_nomes -> validar_emails -> salvar_csv -> END
#    Por fim, devolva o grafo compilado (builder.compile()).
# ---------------------------------------------------------------------------
def construir_grafo():
    """Cria o StateGraph com os 4 nós e devolve o grafo compilado."""
    ## aqui coloque seu código ###


# ---------------------------------------------------------------------------
# 5. ENTRADA PRINCIPAL
# ---------------------------------------------------------------------------
def main() -> None:
    """Resolve os caminhos, invoca o grafo, imprime resumo + preview."""
    aqui = Path(__file__).parent
    entrada = aqui / "cadastro.csv"
    saida = aqui / "cadastro_nova.csv"

    print("=" * 60)
    print("  PIPELINE DE SANITIZAÇÃO — LangGraph")
    print("=" * 60)

    # Passos sugeridos:
    # 1) grafo = construir_grafo()
    # 2) Invoque o grafo com o estado inicial:
    #
    #       {
    #           "caminho_in": str(entrada),
    #           "caminho_out": str(saida),
    #           "linhas": [],
    #           "total": 0,
    #           "logs": [],
    #       }
    #
    #    (opcional) passe config={"tags": [...], "metadata": {...}}
    #    para enviar tags ao LangSmith se o .env estiver configurado.
    #
    # 3) Imprima:
    #       - total processado (estado_final["total"])
    #       - cada log da execução
    #       - preview das 5 primeiras linhas tratadas, no formato:
    #           nome=... -> nome_tratado=...  email_valido=sim/nao
    ## aqui coloque seu código ###


if __name__ == "__main__":
    main()
