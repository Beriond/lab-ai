"""
Exercício 04 — Role Prompting

Definir um papel (persona) muda o tom, o vocabulário e o nível
de profundidade da resposta. No LangChain, isso é uma
`SystemMessage` no início do template.
"""

from __future__ import annotations
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

MODELO = "llama3.2"
PERGUNTA = "O que é uma API REST?"

PAPEIS = {
    "Professor para crianças": (
        "Você é um professor que explica conceitos de tecnologia para "
        "crianças de 10 anos. Use analogias do dia a dia e nada de jargão."
    ),
    "Arquiteto sênior": (
        "Você é um arquiteto de software sênior. Responda de forma técnica, "
        "objetiva, citando trade-offs (idempotência, statelessness, versionamento)."
    ),
    "Poeta": (
        "Você é um poeta brasileiro contemporâneo. Responda em versos livres, "
        "com no máximo 8 linhas."
    ),
}


def perguntar_com_papel(papel: str, pergunta: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{papel}"),
            ("human", "{pergunta}"),
        ]
    )
    llm = ChatOllama(model=MODELO, temperature=0.4)
    cadeia = prompt | llm | StrOutputParser()
    return cadeia.invoke({"papel": papel, "pergunta": pergunta}).strip()


if __name__ == "__main__":
    print(f"Pergunta: {PERGUNTA}\n")
    for nome, papel in PAPEIS.items():
        print("=" * 60)
        print(f"Papel: {nome}")
        print("=" * 60)
        print(perguntar_com_papel(papel, PERGUNTA))
        print()
