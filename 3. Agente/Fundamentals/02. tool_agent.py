"""
Agente simples com uma tool (calculadora)

Objetivo:
    Mostrar o ciclo básico de um agente ReAct com UMA ferramenta,
    sem `response_format`. É o "Hello World" do tool calling:

        Usuário pergunta -> LLM decide chamar tool -> tool roda ->
        LLM lê o resultado -> LLM responde em texto.
"""

from __future__ import annotations
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent

MODELO = "llama3.2"


# ---------------------------------------------------------------------------
# 1. Tool: calculadora simples. A docstring é o que o LLM lê para decidir
#    quando chamar a função e como preencher os argumentos.
# ---------------------------------------------------------------------------
@tool
def calculadora(operacao: str, a: float, b: float) -> str:
    """Faz uma operação aritmética entre dois números.

    Args:
        operacao: uma das strings 'somar', 'subtrair', 'multiplicar', 'dividir'.
        a: primeiro número.
        b: segundo número.

    Returns:
        Texto com o resultado da operação.
    """
    if operacao == "somar":
        resultado = a + b
    elif operacao == "subtrair":
        resultado = a - b
    elif operacao == "multiplicar":
        resultado = a * b
    elif operacao == "dividir":
        if b == 0:
            return "Erro: divisão por zero."
        resultado = a / b
    else:
        return f"Operação desconhecida: {operacao}"
    return f"O resultado de {operacao}({a}, {b}) é {resultado}."


# ---------------------------------------------------------------------------
# 2. Monta o agente. Sem `response_format` — a resposta final é texto.
# ---------------------------------------------------------------------------
system_prompt = (
    "Você é um assistente que ajuda com cálculos.\n"
    "SEMPRE use a ferramenta `calculadora` em vez de calcular de cabeça.\n"
    "No final, devolva o resultado em uma frase curta, em português."
)

llm = ChatOllama(model=MODELO, temperature=0)

agente = create_agent(
    model=llm,
    tools=[calculadora],
    system_prompt=system_prompt,
)


# ---------------------------------------------------------------------------
# 3. Execução com streaming — mostra cada passo do agente.
# ---------------------------------------------------------------------------
def perguntar(texto: str) -> str:
    print(f"\nEu: {texto}")
    print("--- Passos do agente ---")
    estado_final = None
    for estado_final in agente.stream(
        {"messages": [HumanMessage(content=texto)]},
        stream_mode="values",):
        msgs = estado_final.get("messages", [])
        if not msgs:continue
        ult = msgs[-1]
        tipo = type(ult).__name__
        tool_calls = getattr(ult, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                print(f"  [{tipo}] tool_calls -> {tc['name']}({tc['args']})")
        else:
            texto_msg = (getattr(ult, "content", "") or "")[:200]
            print(f"  [{tipo}] {texto_msg}")
    return estado_final["messages"][-1].content


if __name__ == "__main__":
    resposta = perguntar("Quanto é 47 multiplicado por 13?")
    print(f"\nBot: {resposta}")

    resposta = perguntar("E 200 dividido por 8?")
    print(f"\nBot: {resposta}")
