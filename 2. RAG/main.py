"""
main.py — pipeline RAG ponta a ponta + chat interativo.

Fluxo:
    1. Carrega PDFs de ../dados/        (PyPDFLoader)
    2. Quebra em chunks                 (RecursiveCharacterTextSplitter)
    3. Indexa no ChromaDB               (langchain-chroma + OllamaEmbeddings)
    4. Abre um loop de perguntas        (cadeia LCEL)

Como rodar:
    cd "2. RAG"
    python main.py

Para resetar o índice, apague a pasta `chroma_db/` e rode de novo.
"""

from __future__ import annotations

from rag_lib import (
    carregar_pdfs,
    chunkar_documentos,
    gerar_resposta,
    get_vector_store,
    indexar_chunks,
)


def passo_indexacao() -> None:
    vs = get_vector_store()
    qtd = vs._collection.count()  # noqa: SLF001
    if qtd > 0:
        print(f"[indice] Já existe um índice com {qtd} chunk(s). Pulando indexação.")
        print("       (Apague a pasta chroma_db/ se quiser reindexar do zero.)\n")
        return

    print("[1/3] Carregando PDFs...")
    docs = carregar_pdfs()
    if not docs:
        print("\n[ERRO] Nenhum PDF encontrado em ../dados/.")
        print("Coloque ao menos 1 arquivo .pdf nessa pasta e rode de novo.")
        raise SystemExit(1)

    print(f"      → {len(docs)} página(s) extraída(s).")

    print("[2/3] Gerando chunks...")
    chunks = chunkar_documentos(docs)
    print(f"      → {len(chunks)} chunk(s).")

    print("[3/3] Gerando embeddings e indexando no ChromaDB (pode levar um tempo)...")
    indexar_chunks(chunks)

    vs = get_vector_store()
    print(f"\n[ok] Índice pronto com {vs._collection.count()} chunk(s).\n")  # noqa: SLF001


def loop_chat() -> None:
    print("=" * 60)
    print(" CHAT RAG — pergunte sobre os documentos indexados ")
    print(" (digite 'sair' para encerrar)")
    print("=" * 60)

    while True:
        try:
            pergunta = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not pergunta:
            continue
        if pergunta.lower() in {"sair", "exit", "quit"}:
            break

        resultado = gerar_resposta(pergunta, k=4)
        print("\n--- Resposta ---")
        print(resultado["resposta"])
        print("\n--- Fontes usadas ---")
        for i, f in enumerate(resultado["fontes"], 1):
            print(f"  [{i}] {f['fonte']} p.{f['pagina']} (distância={f['distancia']:.3f})")

    print("Até a próxima!")


if __name__ == "__main__":
    passo_indexacao()
    loop_chat()
