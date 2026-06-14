"""
Exercício 03 — Chain-of-Thought (CoT) com streaming

A ideia do CoT é simples: PARA PROBLEMAS QUE EXIGEM RACIOCÍNIO em
múltiplos passos, pedir ao modelo para "pensar em voz alta" antes de
responder costuma melhorar drasticamente a precisão.

Aqui vamos comparar lado a lado, em 3 tipos diferentes de problema:

    1. Aritmética em múltiplos passos
    2. Lógica relacional (quem é mais novo/velho)
    3. Raciocínio sobre datas

Para deixar a cadeia de pensamento VISÍVEL, usamos `llm.stream(...)`:
o modelo vai imprimindo cada pedaço (token) conforme gera. Você
literalmente VÊ o raciocínio sendo construído.
"""

from __future__ import annotations
import re
import sys
import time
from dataclasses import dataclass
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

MODELO = "llama3.2"


@dataclass
class Problema:
    nome: str
    enunciado: str
    resposta_esperada: str


PROBLEMAS = [
    Problema(
        nome="Aritmética em múltiplos passos",
        enunciado=(
            "Maria tem 3 caixas. Em cada caixa há 4 sacos. "
            "Em cada saco há 7 bolinhas. Ela deu 12 bolinhas para o irmão. "
            "Quantas bolinhas ela tem agora?"
        ),
        resposta_esperada="72",
    ),
    Problema(
        nome="Lógica relacional",
        enunciado=(
            "Ana é mais velha que Bia. Carla é mais nova que Bia. "
            "Diana é mais velha que Ana. Quem é a mais nova de todas?"
        ),
        resposta_esperada="Carla",
    ),
    Problema(
        nome="Raciocínio sobre datas",
        enunciado=(
            "Se hoje é quarta-feira, que dia da semana será daqui a 100 dias? "
            "Mostre a divisão por 7."
        ),
        resposta_esperada="sexta",  # 100 % 7 = 2 → quarta + 2 = sexta
    ),
]

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------
TEMPLATE_DIRETO = (
    "{problema}\n\n"
    "Responda diretamente"
)

TEMPLATE_COT = (
    "{problema}\n\n"
    "Pense em voz alta, passo a passo:\n"
    "  Passo 1: identifique os dados do problema.\n"
    "  Passo 2: liste a(s) operação(ões) necessária(s).\n"
    "  Passo 3: faça cada cálculo/raciocínio explicitamente.\n"
    "  Passo 4: conclua.\n\n"
    "Ao final, escreva em uma linha separada exatamente:\n"
    "Resposta final: <sua resposta>"
)


def _construir_cadeia(template: str):
    llm = ChatOllama(model=MODELO, temperature=0.0)
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# Modo "direto" — sem mostrar pensamento
# ---------------------------------------------------------------------------
def responder_direto(enunciado: str) -> str:
    cadeia = _construir_cadeia(TEMPLATE_DIRETO)
    return cadeia.invoke({"problema": enunciado}).strip()


# ---------------------------------------------------------------------------
# Modo CoT — streaming token por token (você VÊ o pensamento)
# ---------------------------------------------------------------------------
def responder_com_cot_streaming(enunciado: str) -> tuple[str, str]:
    """
    Faz streaming do raciocínio em tempo real e retorna (texto_completo, resposta_extraida).
    """
    cadeia = _construir_cadeia(TEMPLATE_COT)

    print("    ┌─ pensamento do modelo (em tempo real) " + "─" * 28)
    print("    │ ", end="", flush=True)

    pedacos: list[str] = []
    for pedaco in cadeia.stream({"problema": enunciado}):
        # Imprime o token e mantém a barra lateral em quebras de linha
        sys.stdout.write(pedaco.replace("\n", "\n    │ "))
        sys.stdout.flush()
        pedacos.append(pedaco)

    print("\n    └" + "─" * 67)

    texto = "".join(pedacos)
    resposta = _extrair_resposta_final(texto)
    return texto, resposta


def _extrair_resposta_final(texto: str) -> str:
    """Pega o que vier depois de 'Resposta final:' (insensível a caixa)."""
    match = re.search(r"resposta\s+final\s*:\s*(.+)", texto, flags=re.IGNORECASE)
    if not match:
        return "(não encontrei 'Resposta final:' no texto)"
    # Pega só a primeira linha após o marcador
    return match.group(1).strip().splitlines()[0].strip()


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------
def _acertou(resposta: str, esperada: str) -> bool:
    return esperada.lower() in resposta.lower()


def comparar_em_um_problema(p: Problema) -> tuple[bool, bool, int]:
    print("\n" + "═" * 72)
    print(f" {p.nome}")
    print("═" * 72)
    print(f" Enunciado: {p.enunciado}")
    print(f" Resposta esperada: {p.resposta_esperada}")

    # ---- direto ----
    print("\n  >> SEM CoT (pedindo resposta direta)")
    t0 = time.perf_counter()
    direto = responder_direto(p.enunciado)
    dur_direto = time.perf_counter() - t0
    print(f"    resposta: {direto!r}")
    print(f"    tempo:    {dur_direto:.2f}s")

    ok_direto = _acertou(direto, p.resposta_esperada)
    print("    " + ("[ACERTOU]" if ok_direto else "[ERROU]"))

    # ---- CoT com streaming ----
    print("\n  >> COM CoT (pensando passo a passo, em streaming)")
    t0 = time.perf_counter()
    texto_cot, resp_cot = responder_com_cot_streaming(p.enunciado)
    dur_cot = time.perf_counter() - t0
    print(f"\n    resposta extraída: {resp_cot!r}")
    print(f"    tamanho do pensamento: {len(texto_cot)} caracteres")
    print(f"    tempo:    {dur_cot:.2f}s")

    ok_cot = _acertou(resp_cot, p.resposta_esperada)
    print("    " + ("[ACERTOU]" if ok_cot else "[ERROU]"))

    return ok_direto, ok_cot, len(texto_cot)


def main() -> None:
    print("Comparação Direto vs. Chain-of-Thought em 3 problemas\n")
    print("Dica: assista o bloco 'pensamento do modelo' aparecer linha a linha.")
    print("Isso é o LLM construindo a resposta um token de cada vez.")

    placar_direto = placar_cot = 0
    total_chars_cot = 0

    for p in PROBLEMAS:
        ok_d, ok_c, chars = comparar_em_um_problema(p)
        placar_direto += int(ok_d)
        placar_cot += int(ok_c)
        total_chars_cot += chars

    print("\n" + "═" * 72)
    print(" Resumo final")
    print("═" * 72)
    print(f"  Direto: {placar_direto}/{len(PROBLEMAS)} acertos")
    print(f"  CoT:    {placar_cot}/{len(PROBLEMAS)} acertos "
          f"(usou {total_chars_cot} caracteres a mais 'pensando')")
    print()
    print("Conclusão didática:")
    print("  - CoT gasta mais tokens (logo, mais latência e custo).")
    print("  - Em troca, acerta com mais frequência em problemas que exigem")
    print("    múltiplos passos. Use CoT quando o erro custar mais caro que")
    print("    a latência. Para tarefas simples, prefira respostas diretas.")


if __name__ == "__main__":
    main()
