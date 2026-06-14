"""
Arestas condicionais (roteamento)

Até agora as arestas eram FIXAS: sempre o mesmo caminho. Aqui o
grafo decide para ONDE ir em tempo de execução, com base no estado.

Quem decide? Uma função normal (`router`) que recebe o estado e
devolve o nome do próximo nó.

Fluxo:
                          ┌── "especialista" → END
    analisar ──router()──►├── "marketing"    → END
                          └── "suporte"      → END
"""

from __future__ import annotations
import operator
from typing import Annotated, List, Literal, TypedDict, Optional
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


# Schema de roteamento — o LLM preenche e o router olha.
class Roteamento(BaseModel):
    categoria: str = Field(description="'Bug', 'Elogio' ou 'Outro'")
    sentimento: str = Field(description="'Positivo' ou 'Negativo'")
    justificativa: str = Field(description="Breve explicação da decisão")


class Estado(TypedDict):
    cliente: str
    feedback: str
    decisao: Optional[Roteamento]
    equipe: Optional[str]
    resposta: Optional[str]
    historico: Annotated[List[str], operator.add]


llm = ChatOllama(model=MODELO, temperature=0)


# Nós
def no_analisar(state: Estado) -> dict:
    print(f"[ANALISAR] classificando feedback de {state['cliente']}")
    decisao = llm.with_structured_output(Roteamento).invoke(
        f"Classifique este feedback: {state['feedback']}"
    )
    print(f"  categoria={decisao.categoria} sentimento={decisao.sentimento}")
    return {
        "decisao": decisao,
        "historico": [f"classificado como {decisao.categoria}"],
    }


def no_especialista(state: Estado) -> dict:
    print("[ESPECIALISTA] roteando para engenharia")
    return {
        "equipe": "Engenharia / QA",
        "resposta": "Obrigado por reportar. Nossa equipe técnica está investigando.",
        "historico": ["tratado pela engenharia"],
    }


def no_marketing(state: Estado) -> dict:
    print("[MARKETING] roteando para marketing")
    return {
        "equipe": "Marketing",
        "resposta": "Que ótimo! Vamos compartilhar com todo o time.",
        "historico": ["tratado pelo marketing"],
    }


def no_suporte(state: Estado) -> dict:
    print("[SUPORTE] roteando para suporte N1")
    return {
        "equipe": "Suporte N1",
        "resposta": "Recebemos sua mensagem; um atendente entrará em contato.",
        "historico": ["tratado pelo suporte"],
    }


# Router: função simples (sem LLM!) que escolhe o próximo nó.
# O retorno DEVE bater com as chaves do dicionário em
# `add_conditional_edges`.
def router(state: Estado) -> Literal["especialista", "marketing", "suporte"]:
    d = state["decisao"]
    if d.categoria == "Bug":
        return "especialista"
    if d.sentimento == "Positivo":
        return "marketing"
    return "suporte"


# Montagem do grafo
builder = StateGraph(Estado)
builder.add_node("analisar", no_analisar)
builder.add_node("especialista", no_especialista)
builder.add_node("marketing", no_marketing)
builder.add_node("suporte", no_suporte)

builder.add_edge(START, "analisar")

builder.add_conditional_edges(
    "analisar",
    router,
    {
        "especialista": "especialista",
        "marketing": "marketing",
        "suporte": "suporte",
    },
)

builder.add_edge("especialista", END)
builder.add_edge("marketing", END)
builder.add_edge("suporte", END)

grafo = builder.compile()


# Execução com 3 casos diferentes — cada um deve cair em uma rota
if __name__ == "__main__":
    casos = [
        ("Alice", "Encontrei um erro crítico na tela de pagamentos."),
        ("Bruno", "Adorei a nova interface, ficou muito mais rápida!"),
        ("Carla", "Como faço para trocar minha senha?"),
    ]

    for nome, msg in casos:
        print("\n" + "=" * 60)
        print(f"Cliente: {nome} | Mensagem: {msg}")
        print("=" * 60)
        # Um trace por caso — no painel dá pra filtrar pelo nome do cliente
        config = {
            "tags": ["aula:05", "demo:conditional"],
            "metadata": {"aula": "05", "topico": "router", "cliente": nome},
        }
        resultado = grafo.invoke(
            {"cliente": nome, "feedback": msg, "historico": []},
            config,
        )
        print(f"→ Equipe : {resultado['equipe']}")
        print(f"→ Resposta: {resultado['resposta']}")
