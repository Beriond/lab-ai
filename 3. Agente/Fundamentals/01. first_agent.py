"""
Primeiro agente com saída estruturada

O que é um agente?
    Um agente é um LLM que recebe um objetivo, decide o que fazer e
    pode usar ferramentas. Aqui ainda não usamos ferramentas — vamos
    apenas montar o agente mais simples possível usando o utilitário
    `create_react_agent` do LangGraph.

Saída estruturada:
    Em vez de receber um texto livre, pedimos ao modelo que responda
    em um formato fixo descrito por um schema do Pydantic. Isso é
    útil quando o resultado vai alimentar outro sistema (banco, API).
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

MODELO = "llama3.2"


# Schema da resposta — o agente é OBRIGADO a preencher esses campos
class InfoCapital(BaseModel):
    cidade: str = Field(description="Nome da capital")
    pais: str = Field(description="País a que a capital pertence")
    curiosidade: str = Field(description="Uma curiosidade turística da cidade")


llm = ChatOllama(model=MODELO, temperature=0)

agente = create_agent(
    model=llm,
    tools=[],  
    system_prompt="Você é um especialista em turismo. Responda em português.",
    response_format=InfoCapital,
)


def perguntar(pergunta: str) -> InfoCapital:
    """Envia uma pergunta ao agente, mostra cada passo e devolve o objeto estruturado."""
    
    estado_final = None
    print("--- Passos do agente ---")

    for estado_final in agente.stream(
        {"messages": [HumanMessage(content=pergunta)]},
        stream_mode="values",):
        msgs = estado_final.get("messages", [])
        if not msgs:continue
        ult = msgs[-1]
        tipo = type(ult).__name__
        tool_calls = getattr(ult, "tool_calls", None)
        if tool_calls:
            nomes = [t["name"] for t in tool_calls]
            print(f"  [{tipo}] tool_calls -> {nomes}")
        else:
            texto = (getattr(ult, "content", "") or "")[:140]
            print(f"  [{tipo}] {texto}")
    return estado_final["structured_response"]


if __name__ == "__main__":
    info = perguntar("Qual é a capital da França?")
    print("\n=== Resposta estruturada do agente ===")
    print(f"Cidade.......: {info.cidade}")
    print(f"País.........: {info.pais}")
    print(f"Curiosidade..: {info.curiosidade}")
