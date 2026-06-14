"""
Nós (Nodes) e Estado (State) no LangGraph

Ideia central:
    Um grafo do LangGraph é só um "fluxograma" onde cada CAIXA é um
    nó (função Python) e cada SETA é uma aresta. Todos os nós leem
    e escrevem em um dicionário compartilhado chamado `State`.

O que este exemplo faz:
    Recebe um feedback de cliente, limpa o texto, pede ao LLM que
    analise sentimento + urgência e finaliza o fluxo.

    [START] → sanitizar → analisar_llm → concluir → [END]
"""

from __future__ import annotations
import json
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


# Schema da análise feita pelo LLM (saída estruturada)
class AnaliseFeedback(BaseModel):
    sentimento: str = Field(description="'Positivo', 'Negativo' ou 'Neutro'")
    urgencia: str = Field(description="'Baixa', 'Média' ou 'Alta'")
    resumo: str = Field(description="Resumo em uma frase")


# Estado: o "dicionário" que passa entre os nós.
# Annotated[..., operator.add] = quando dois nós escrevem nesse
# campo, o LangGraph faz append em vez de sobrescrever.
class Estado(TypedDict):
    cliente: str
    feedback_bruto: str
    feedback_limpo: Optional[str]
    analise: Optional[AnaliseFeedback]
    historico: Annotated[List[str], operator.add]


llm = ChatOllama(model=MODELO, temperature=0)


# Os nós são funções: recebem o estado e devolvem um dict com os
# campos a atualizar.
def no_sanitizar(state: Estado) -> dict:
    print(f"[SANITIZAR] cliente: {state['cliente']}")
    return {
        "feedback_limpo": state["feedback_bruto"].strip(),
        "historico": ["texto sanitizado"],
    }


def no_analisar(state: Estado) -> dict:
    print("[ANALISAR] rodando LLM com saída estruturada")
    prompt = (
        f"Analise o feedback do cliente {state['cliente']}:\n"
        f"\"{state['feedback_limpo']}\""
    )
    analise = llm.with_structured_output(AnaliseFeedback).invoke(prompt)
    print(f"  sentimento={analise.sentimento} urgencia={analise.urgencia}")
    return {
        "analise": analise,
        "historico": ["análise concluída"],
    }


def no_concluir(state: Estado) -> dict:
    print("[CONCLUIR] workflow finalizado")
    return {"historico": ["workflow finalizado"]}


# Montagem do grafo: adiciona nós e ligações.
builder = StateGraph(Estado)
builder.add_node("sanitizar", no_sanitizar)
builder.add_node("analisar_llm", no_analisar)
builder.add_node("concluir", no_concluir)

builder.add_edge(START, "sanitizar")
builder.add_edge("sanitizar", "analisar_llm")
builder.add_edge("analisar_llm", "concluir")
builder.add_edge("concluir", END)

grafo = builder.compile()

# Execução
if __name__ == "__main__":
    entrada = {
        "cliente": "Maria Silva",
        "feedback_bruto": (
            "  Nossa, parabéns pelo serviço. O sistema caiu 3 vezes hoje e "
            "perdi todo meu trabalho. Simplesmente incrível.  "
        ),
        "historico": [],
    }

    # Tags e metadata aparecem como filtros no LangSmith
    config = {
        "tags": ["aula:03", "demo:nodes-and-state"],
        "metadata": {"aula": "03", "topico": "nodes_state", "cliente": entrada["cliente"]},
    }
    resultado = grafo.invoke(entrada, config)

    print("\n=== Estado final ===")
    # converte o pydantic em dict só para imprimir bonito
    resultado["analise"] = resultado["analise"].model_dump()
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
