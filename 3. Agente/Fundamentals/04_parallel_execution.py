"""
Execução paralela (Fan-out / Fan-in)

Por que paralelizar?
    Quando temos várias análises INDEPENDENTES sobre o mesmo dado,
    rodar uma de cada vez desperdiça tempo. No LangGraph basta
    apontar várias arestas saindo do MESMO nó — elas executam em
    paralelo. Para "juntar" os resultados, todas voltam para um
    único nó agregador.

Fluxo:
                 ┌──→ sentimento ─┐
    sanitizar ───┼──→ categoria  ─┼──→ agregador → END
                 └──→ urgencia   ─┘
"""

from __future__ import annotations
import operator
from typing import Annotated, List, TypedDict, Optional
from pydantic import BaseModel, Field

# Carrega .env da raiz do projeto para ativar o tracing do LangSmith
try:
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph

MODELO = "llama3.2"


# Cada análise paralela tem seu próprio schema
class Sentimento(BaseModel):
    sentimento: str = Field(description="'Positivo', 'Negativo' ou 'Neutro'")


class Categoria(BaseModel):
    categoria: str = Field(description="'Bug', 'Financeiro', 'Sugestão' ou 'Suporte'")


class Urgencia(BaseModel):
    urgencia: str = Field(description="'Baixa', 'Média' ou 'Alta'")


# Estado compartilhado.
# `historico` usa `operator.add`: cada nó faz APPEND nele.
class Estado(TypedDict):
    cliente: str
    feedback: str
    res_sentimento: Optional[str]
    res_categoria: Optional[str]
    res_urgencia: Optional[str]
    analise_final: Optional[str]
    historico: Annotated[List[str], operator.add]


llm = ChatOllama(model=MODELO, temperature=0)


# Nós
def no_sanitizar(state: Estado) -> dict:
    print(f"[SANITIZAR] cliente: {state['cliente']}")
    return {"historico": ["sanitização ok"]}


def no_sentimento(state: Estado) -> dict:
    print("[SENTIMENTO] rodando LLM com saída estruturada")
    r = llm.with_structured_output(Sentimento).invoke(
        f"Qual o sentimento? Feedback: {state['feedback']}"
    )
    return {
        "res_sentimento": r.sentimento,
        "historico": [f"sentimento={r.sentimento}"],
    }


def no_categoria(state: Estado) -> dict:
    print("[CATEGORIA] rodando LLM com saída estruturada")
    r = llm.with_structured_output(Categoria).invoke(
        f"Categorize este feedback: {state['feedback']}"
    )
    return {
        "res_categoria": r.categoria,
        "historico": [f"categoria={r.categoria}"],
    }


def no_urgencia(state: Estado) -> dict:
    print("[URGENCIA] rodando LLM com saída estruturada")
    r = llm.with_structured_output(Urgencia).invoke(
        f"Qual a urgência deste caso? Feedback: {state['feedback']}"
    )
    return {
        "res_urgencia": r.urgencia,
        "historico": [f"urgencia={r.urgencia}"],
    }


def no_agregador(state: Estado) -> dict:
    """Fan-in: o LangGraph só roda este nó depois que TODOS os três
    nós paralelos terminarem."""
    print("[AGREGADOR] consolidando resultados")
    resumo = (
        f"Sentimento: {state['res_sentimento']} | "
        f"Categoria: {state['res_categoria']} | "
        f"Urgência: {state['res_urgencia']}"
    )
    return {"analise_final": resumo, "historico": ["agregado"]}


# Montagem do grafo
builder = StateGraph(Estado)
builder.add_node("sanitizar", no_sanitizar)
builder.add_node("sentimento", no_sentimento)
builder.add_node("categoria", no_categoria)
builder.add_node("urgencia", no_urgencia)
builder.add_node("agregador", no_agregador)

# 3 arestas saindo do mesmo nó = paralelo
builder.add_edge(START, "sanitizar")
builder.add_edge("sanitizar", "sentimento")
builder.add_edge("sanitizar", "categoria")
builder.add_edge("sanitizar", "urgencia")

# 3 arestas chegando no mesmo nó = espera todas
builder.add_edge("sentimento", "agregador")
builder.add_edge("categoria", "agregador")
builder.add_edge("urgencia", "agregador")
builder.add_edge("agregador", END)

grafo = builder.compile()


# Execução
if __name__ == "__main__":
    entrada = {
        "cliente": "João Silva",
        "feedback": (
            "Paguei o boleto há 4 dias e ainda não liberaram meu acesso. "
            "Preciso disso agora!"
        ),
        "historico": [],
    }

    # No painel do LangSmith dá pra ver as 3 chamadas paralelas
    # acontecendo no mesmo nível da árvore, com latência sobreposta.
    config = {
        "tags": ["aula:04", "demo:parallel"],
        "metadata": {"aula": "04", "topico": "fan_out_fan_in", "cliente": entrada["cliente"]},
    }
    resultado = grafo.invoke(entrada, config)

    print("\n=== Resumo final ===")
    print(resultado["analise_final"])

    print("\n=== Ordem em que os nós terminaram ===")
    for i, passo in enumerate(resultado["historico"], 1):
        print(f"  {i}. {passo}")
