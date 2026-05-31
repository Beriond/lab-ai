"""
Etapa 5 do RAG — Retrieval (busca por similaridade)

`Chroma.similarity_search_with_score(pergunta, k=4)` faz tudo:
    1. Gera o embedding da pergunta (via `embedding_function`).
    2. Compara com todos os vetores indexados (cosseno).
    3. Retorna os k mais próximos com sua distância.

A `Chroma.as_retriever()` devolve um objeto Retriever que pode ser
encadeado em LCEL — usaremos isso no passo 6.
"""

from __future__ import annotations

from pathlib import Path

from rag_lib import get_vector_store, recuperar


def main() -> None:
    vs = get_vector_store()
    if vs._collection.count() == 0:  # noqa: SLF001
        print("A coleção está vazia. Rode primeiro: python 04_indexacao.py")
        return

    pergunta = "Qual o tema principal do material que indexei?"
    print(f">> Pergunta de exemplo: {pergunta}\n")

    resultados = recuperar(pergunta, k=4)

    for i, (doc, distancia) in enumerate(resultados, 1):
        fonte = Path(doc.metadata.get("source", "?")).name
        pagina = doc.metadata.get("page", "?")
        print(f"--- Top {i} — distância={distancia:.4f} ---")
        print(f"Fonte: {fonte} (página {pagina})")
        texto = doc.page_content
        print(texto[:300] + ("…" if len(texto) > 300 else ""))
        print()


if __name__ == "__main__":
    main()
