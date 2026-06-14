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
from importlib import import_module
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import OVERLAP, TAMANHO_CHUNK

carregar_pdfs = import_module("01_carregamento").carregar_pdfs


def chunkar_documentos(
    documentos: list[Document],
    tamanho: int = TAMANHO_CHUNK,
    overlap: int = OVERLAP,
) -> list[Document]:
    """
    Quebra cada `Document` em chunks menores, preservando os metadados.
    Usa o splitter recursivo (tenta separar primeiro por parágrafo,
    depois linha, depois espaço).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=tamanho,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documentos)


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
