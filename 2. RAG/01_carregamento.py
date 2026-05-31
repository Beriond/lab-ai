"""
Etapa 1 do RAG — Carregamento (com PyPDFLoader do LangChain)

Lê todos os PDFs da pasta `dados/` (recursivamente) e devolve uma lista
de `langchain_core.documents.Document` — 1 por página de PDF.

Cada `Document` tem:
    - page_content: o texto extraído
    - metadata: dict com 'source' (caminho do arquivo) e 'page' (índice da página)
"""

from __future__ import annotations

from rag_lib import PASTA_DADOS, carregar_pdfs


def main() -> None:
    print(f"Lendo PDFs de: {PASTA_DADOS}\n")
    docs = carregar_pdfs()

    if not docs:
        print("[AVISO] Nenhum PDF encontrado. Coloque arquivos .pdf na pasta dados/.")
        return

    fontes = sorted({d.metadata.get("source", "?") for d in docs})
    print(f"[ok] {len(docs)} página(s) extraída(s) de {len(fontes)} arquivo(s):")
    for f in fontes:
        n = sum(1 for d in docs if d.metadata.get("source") == f)
        print(f"  - {f}: {n} página(s)")

    print("\n--- Prévia do primeiro Document ---")
    primeiro = docs[0]
    print(f"metadata: {primeiro.metadata}")
    conteudo = primeiro.page_content
    print(conteudo[:500] + ("…" if len(conteudo) > 500 else ""))


if __name__ == "__main__":
    main()
