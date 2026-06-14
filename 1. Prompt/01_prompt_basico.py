"""
Exercício 01 — Seu primeiro prompt com LangChain

Aqui você vai usar o cliente `ChatOllama` do `langchain-ollama` para
falar com o Llama 3.2 localmente. Veja como pequenas mudanças no
prompt e na `temperature` afetam a resposta.

Mexa nos textos e rode de novo!
"""

from __future__ import annotations
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

MODELO = "llama3.2"

def chamar_llm(prompt: str, temperature: float = 0.7) -> str:
    """Cria um ChatOllama e devolve o texto da resposta."""
    llm = ChatOllama(model=MODELO, temperature=temperature)
    resposta = llm.invoke(prompt)
    return resposta.content


def demo_prompt_vago_vs_especifico() -> None:
    print("=" * 60)
    print("Demo 1: Prompt vago vs. prompt específico")
    print("=" * 60)

    vago = "Me fale sobre cachorros."
    especifico = (
        "Liste 3 raças de cachorro recomendadas para apartamentos pequenos. "
        "Para cada uma, dê o porte, o nível de energia e uma curiosidade. "
        "Responda em Markdown com bullets."
    )

    print("\n>> Prompt vago:")
    print(vago)
    print("\n>> Resposta:")
    print(chamar_llm(vago))

    print("\n>> Prompt específico:")
    print(especifico)
    print("\n>> Resposta:")
    print(chamar_llm(especifico))


def demo_temperature() -> None:
    print("\n" + "=" * 60)
    print("Demo 2: Como a temperature muda a resposta")
    print("=" * 60)

    prompt = "Sugira um nome criativo para uma cafeteria temática de programadores."

    for temp in (0.0, 0.7, 1.2):
        print(f"\n>> temperature={temp}")
        print(chamar_llm(prompt, temperature=temp))


if __name__ == "__main__":
    print("rodando")
    demo_prompt_vago_vs_especifico()
    demo_temperature()
