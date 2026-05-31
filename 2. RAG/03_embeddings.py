"""
Etapa 3 do RAG — Embeddings (OllamaEmbeddings via LangChain)

`OllamaEmbeddings` é o wrapper LangChain para o endpoint de embeddings
do Ollama. Ele expõe dois métodos principais:

    - embed_query(texto)        → vetor único (use no momento da busca)
    - embed_documents([textos]) → lista de vetores (use ao indexar)

Vamos comparar a similaridade entre frases para mostrar que o
significado é capturado, não só palavras.
"""

from __future__ import annotations

from math import sqrt

from rag_lib import get_embeddings


def cosseno(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (sqrt(sum(x * x for x in a)) * sqrt(sum(y * y for y in b)))


def main() -> None:
    embeddings = get_embeddings()

    frases = [
        "O cachorro está latindo no quintal.",
        "Um cão faz barulho no jardim.",
        "A bolsa de valores caiu hoje.",
    ]

    vetores = embeddings.embed_documents(frases)
    print(f"Gerei {len(vetores)} embedding(s), cada um com {len(vetores[0])} dimensões.\n")

    print("Similaridade do cosseno (1.0 = idêntico, 0.0 = sem relação):")
    print(f"  frase 0 vs frase 1 (dois cães):     {cosseno(vetores[0], vetores[1]):.3f}")
    print(f"  frase 0 vs frase 2 (cão x bolsa):   {cosseno(vetores[0], vetores[2]):.3f}")
    print("\nNote: as duas frases sobre cão ficam bem mais próximas no espaço vetorial.")

    print("\n--- embed_query: usado na hora da busca ---")
    vetor = embeddings.embed_query("teste")
    print(f"dimensão = {len(vetor)} | primeiros 5 valores: {vetor[:5]}")


if __name__ == "__main__":
    main()
