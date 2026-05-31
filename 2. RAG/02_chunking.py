"""
Etapa 2 do RAG — Chunking (RecursiveCharacterTextSplitter)

`RecursiveCharacterTextSplitter` tenta quebrar o texto na ordem:
    1. parágrafo (\\n\\n)
    2. linha (\\n)
    3. frase (". ")
    4. palavra (" ")
    5. caractere

Isso preserva contexto semântico melhor que um corte cego por tamanho.
"""

from __future__ import annotations

from langchain_core.documents import Document

from rag_lib import carregar_pdfs, chunkar_documentos


def demo_em_texto_inline() -> None:
    doc_demo = Document(
        page_content=(
            "RAG significa Retrieval-Augmented Generation. "
            "Ele combina recuperação de informação com geração de linguagem natural. "
            "A ideia é dar ao LLM acesso a uma base externa, para que ele responda "
            "com base em documentos reais e não em sua memória interna. "
        ) * 5,
        metadata={"source": "demo.txt", "page": 0},
    )

    chunks = chunkar_documentos([doc_demo], tamanho=200, overlap=40)
    print(f"Texto demo: {len(doc_demo.page_content)} caracteres → {len(chunks)} chunk(s).\n")
    for i, c in enumerate(chunks):
        print(f"--- chunk {i} ({len(c.page_content)} chars) ---")
        print(c.page_content)
        print()


def demo_em_pdfs() -> None:
    docs = carregar_pdfs()
    if not docs:
        print("[INFO] Sem PDFs em dados/. Pulei a demo nos PDFs.")
        return

    chunks = chunkar_documentos(docs)
    print(f"Gerei {len(chunks)} chunk(s) a partir de {len(docs)} páginas.\n")
    print("Exemplo do primeiro chunk:")
    primeiro = chunks[0]
    print(f"  metadata: {primeiro.metadata}")
    print(f"  tamanho:  {len(primeiro.page_content)} chars")
    print(f"  texto:    {primeiro.page_content[:300]}…")


if __name__ == "__main__":
    print("=== Demo 1 — chunking de um texto inline ===\n")
    demo_em_texto_inline()
    print("\n=== Demo 2 — chunking dos seus PDFs ===\n")
    demo_em_pdfs()
