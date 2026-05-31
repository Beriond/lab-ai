"""
Etapa 4 do RAG — Indexação no ChromaDB (via langchain-chroma)

`Chroma` (do langchain-chroma) é um vector store que:
    - persiste em disco automaticamente (`persist_directory`)
    - aceita uma `embedding_function` (no nosso caso, `OllamaEmbeddings`)
    - gera os embeddings sozinho quando você chama `add_documents`

Idempotência: cada `add_documents` gera ids novos. Para reindexar do
zero, apague a pasta `chroma_db/`.
"""

from __future__ import annotations

from rag_lib import carregar_pdfs, chunkar_documentos, get_vector_store, indexar_chunks


def main() -> None:
    print("[1/3] Carregando PDFs da pasta dados/...")
    docs = carregar_pdfs()
    if not docs:
        print("[ERRO] Nenhum PDF em dados/. Coloque ao menos 1 .pdf e rode de novo.")
        return

    print(f"      → {len(docs)} página(s).")

    print("[2/3] Quebrando em chunks...")
    chunks = chunkar_documentos(docs)
    print(f"      → {len(chunks)} chunk(s).")

    print("[3/3] Indexando no ChromaDB (gera embeddings via Ollama)...")
    indexar_chunks(chunks)

    vs = get_vector_store()
    print(f"\n[ok] Coleção atual contém {vs._collection.count()} chunk(s).")  # noqa: SLF001


if __name__ == "__main__":
    main()
