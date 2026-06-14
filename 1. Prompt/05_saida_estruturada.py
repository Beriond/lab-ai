"""
Exercício 05 — Saída estruturada (Pydantic + with_structured_output)

Quando o resultado do LLM vai alimentar outro pedaço de código,
você precisa de saída previsível. No LangChain, declaramos um
schema em **Pydantic** e usamos `llm.with_structured_output(Schema)`.

A engine cuida de pedir JSON, parsear, validar e devolver
diretamente uma instância do modelo. Sem precisar do `json.loads`.
"""

from __future__ import annotations
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

MODELO = "llama3.2"

TEXTO = (
    "João Silva, 34 anos, trabalha como engenheiro de dados na empresa Acme. "
    "Mora em Belo Horizonte e tem dois cachorros, o Rex e a Luna."
)

class Pessoa(BaseModel):
    """Pessoa mencionada num texto livre."""

    nome: str = Field(description="Nome completo")
    idade: int = Field(description="Idade em anos")
    profissao: str
    empresa: str
    cidade: str
    pets: list[str] = Field(default_factory=list, description="Nomes dos animais de estimação")


def extrair_pessoa(texto: str) -> Pessoa:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Você extrai informações estruturadas a partir de textos em português."),
            ("human", 'Extraia as informações da pessoa descrita no texto:\n\n"""{texto}"""'),
        ]
    )
    llm = ChatOllama(model=MODELO, temperature=0.0)
    llm_estruturado = llm.with_structured_output(Pessoa)
    cadeia = prompt | llm_estruturado
    return cadeia.invoke({"texto": texto})


if __name__ == "__main__":
    print("Texto de entrada:")
    print(f"  {TEXTO}\n")

    pessoa = extrair_pessoa(TEXTO)

    print("Instância de Pydantic retornada:")
    print(pessoa.model_dump_json(indent=2))
    print()

    print("Agora dá pra usar como objeto Python:")
    print(f"  Nome:     {pessoa.nome}")
    print(f"  Idade:    {pessoa.idade}")
    print(f"  Cidade:   {pessoa.cidade}")
    print(f"  Pets:     {', '.join(pessoa.pets) if pessoa.pets else '(nenhum)'}")
