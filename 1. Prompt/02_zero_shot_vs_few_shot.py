"""
Exercício 02 — Zero-shot vs. Few-shot

Por que essa comparação importa?

Zero-shot funciona bem quando a tarefa é genérica (ex.: "este texto é
positivo ou negativo?"). Mas QUEBRA quando a tarefa exige um vocabulário
PRÓPRIO da empresa — categorias internas, formatos custom, etc.

Aqui simulamos um sistema de tickets da empresa "Acme" com 5 categorias
INVENTADAS:

    BUG_CRITICO   — sistema fora do ar / perda de dados
    BUG_MENOR     — quebra mas tem workaround
    MELHORIA      — pedido de funcionalidade nova
    DUVIDA        — pergunta de uso
    ELOGIO        — feedback positivo

O modelo nunca viu essas tags. Veja o que acontece:

    >> Zero-shot: ele INVENTA categorias (BUG, URGENTE, QUESTÃO…).
    >> Few-shot:  ele aprende o "dialeto" da Acme pelos exemplos e
                  responde exatamente uma das 5 tags válidas.
"""

from __future__ import annotations
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

MODELO = "llama3.2"

# Categorias do "Sistema de Tickets da Acme" — totalmente inventadas
CATEGORIAS_VALIDAS = {"BUG_CRITICO", "BUG_MENOR", "MELHORIA", "DUVIDA", "ELOGIO"}

# Tickets + categoria esperada (gabarito, para medir acerto no final)
TICKETS = [
    ("O sistema travou e perdi o relatório que estava fazendo há 2 horas. "
     "Tenho que entregar hoje!", "BUG_CRITICO"),
    ("Adoraria poder exportar os relatórios em PDF além de Excel.", "MELHORIA"),
    ("Como faço para resetar minha senha? Não achei a opção no menu.", "DUVIDA"),
    ("O botão 'Salvar' fica fora da tela em monitores 4K, mas dá pra apertar TAB+Enter.",
     "BUG_MENOR"),
    ("Parabéns pela última atualização! A nova tela está muito mais rápida.", "ELOGIO"),
]


def _construir_cadeia(template: str):
    llm = ChatOllama(model=MODELO, temperature=0.0)
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# Zero-shot — só descrevemos a tarefa, SEM mostrar como é o formato
# ---------------------------------------------------------------------------
TEMPLATE_ZERO_SHOT = (
    "Classifique o ticket de suporte abaixo em uma categoria.\n"
    "Responda APENAS com a categoria (uma palavra).\n\n"
    'Ticket: "{ticket}"\n'
    "Categoria:"
)


def classificar_zero_shot(ticket: str) -> str:
    cadeia = _construir_cadeia(TEMPLATE_ZERO_SHOT)
    return cadeia.invoke({"ticket": ticket}).strip().upper() # está esperando uma string


# ---------------------------------------------------------------------------
# Few-shot — damos 5 exemplos do "dialeto" da Acme
# ---------------------------------------------------------------------------
TEMPLATE_FEW_SHOT = (
    "Você classifica tickets do sistema da Acme nas categorias:\n"
    "BUG_CRITICO, BUG_MENOR, MELHORIA, DUVIDA, ELOGIO.\n"
    "Responda APENAS com a categoria exata.\n\n"
    'Ticket: "Apaguei sem querer um cliente e não tem como desfazer, perdi todo o histórico!"\n'
    "Categoria: BUG_CRITICO\n\n"
    'Ticket: "Seria legal ter um modo escuro na interface."\n'
    "Categoria: MELHORIA\n\n"
    'Ticket: "Onde eu encontro o relatório mensal de vendas?"\n'
    "Categoria: DUVIDA\n\n"
    'Ticket: "Os ícones ficam sobrepostos no Firefox, mas no Chrome funciona normal."\n'
    "Categoria: BUG_MENOR\n\n"
    'Ticket: "A equipe de suporte foi excelente, problema resolvido em 5 minutos!"\n'
    "Categoria: ELOGIO\n\n"
    'Ticket: "{ticket}"\n'
    "Categoria:"
)


def classificar_few_shot(ticket: str) -> str:
    cadeia = _construir_cadeia(TEMPLATE_FEW_SHOT)
    return cadeia.invoke({"ticket": ticket}).strip().upper()


# ---------------------------------------------------------------------------
# Comparação lado a lado
# ---------------------------------------------------------------------------
def _formatar(resposta: str, esperado: str) -> str:
    valida = resposta in CATEGORIAS_VALIDAS
    correta = resposta == esperado
    if correta:
        marca = "[ACERTOU]"
    elif valida:
        marca = "[VALIDA mas errou]"
    else:
        marca = "[FORA DO VOCABULARIO]"
    return f"{marca:>22}  resposta={resposta!r}"


def main() -> None:
    print("=" * 72)
    print(" Classificação de tickets da Acme — 5 categorias inventadas ")
    print(f" {sorted(CATEGORIAS_VALIDAS)}")
    print("=" * 72)

    acertos_zs = acertos_fs = 0
    no_vocab_zs = no_vocab_fs = 0

    for i, (ticket, esperado) in enumerate(TICKETS, 1):
        print(f"\n--- Ticket {i} (esperado: {esperado}) ---")
        print(f'  "{ticket}"\n')

        r_zs = classificar_zero_shot(ticket)
        r_fs = classificar_few_shot(ticket)

        print(f"  Zero-shot {_formatar(r_zs, esperado)}")
        print(f"  Few-shot  {_formatar(r_fs, esperado)}")

        acertos_zs += r_zs == esperado
        acertos_fs += r_fs == esperado
        no_vocab_zs += r_zs not in CATEGORIAS_VALIDAS
        no_vocab_fs += r_fs not in CATEGORIAS_VALIDAS

    total = len(TICKETS)
    print("\n" + "=" * 72)
    print(" Resumo")
    print("=" * 72)
    print(f"  Zero-shot:  {acertos_zs}/{total} acertos | "
          f"{no_vocab_zs}/{total} respostas FORA do vocabulário")
    print(f"  Few-shot:   {acertos_fs}/{total} acertos | "
          f"{no_vocab_fs}/{total} respostas FORA do vocabulário")
    print()
    print("Conclusão didática:")
    print("  - Zero-shot tende a INVENTAR categorias (BUG, URGENTE, QUESTAO…),")
    print("    porque o modelo não conhece o vocabulário interno da Acme.")
    print("  - Few-shot ancora o modelo no formato que VOCÊ definiu, mostrando")
    print("    pelos exemplos como uma resposta válida se parece.")


if __name__ == "__main__":
    main()
