"""
Etapa 4 do RAG — Retrieval (busca por similaridade)

`Chroma.similarity_search_with_score(pergunta, k=4)` faz tudo:
    1. Gera o embedding da pergunta (via `embedding_function`).
    2. Compara com todos os vetores indexados (cosseno).
    3. Retorna os k mais próximos com sua distância.
"""

from __future__ import annotations
from pathlib import Path
from langchain_core.documents import Document
from utils import get_vector_store


def recuperar(pergunta: str, k: int = 4) -> list[tuple[Document, float]]:
    """Top-k chunks mais próximos da pergunta, com a distância (cosseno)."""
    vs = get_vector_store()
    return vs.similarity_search_with_score(pergunta, k=k)


def get_retriever(k: int = 4):
    """Retorna um Retriever pronto para usar em uma cadeia LCEL."""
    return get_vector_store().as_retriever(search_kwargs={"k": k})


def main() -> None:
    vs = get_vector_store()
    if vs._collection.count() == 0:  
        print("A coleção está vazia. Rode primeiro: python 03_indexacao.py")
        return

    pergunta = "Qual o tema principal do material que eu indexei?"
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
