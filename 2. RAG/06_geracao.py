"""
Etapa 6 do RAG — Geração aumentada (cadeia LCEL)

A cadeia canônica de RAG escrita em LCEL:

    {"pergunta": ...}
        ↓
    {"contexto": retriever | formatar_contexto,
     "pergunta": passthrough}
        ↓
    ChatPromptTemplate
        ↓
    ChatOllama
        ↓
    StrOutputParser
        ↓
    string final

Cada `|` é um Runnable do LangChain. Em LangSmith essa cadeia aparece
como uma árvore de spans, e você consegue inspecionar entrada/saída
de cada etapa.
"""

from __future__ import annotations

from rag_lib import construir_cadeia_rag, gerar_resposta, get_vector_store


def main() -> None:
    vs = get_vector_store()
    if vs._collection.count() == 0:  # noqa: SLF001
        print("A coleção está vazia. Rode primeiro: python 04_indexacao.py")
        return

    pergunta = "Faça um resumo de 3 frases sobre o material que indexei."
    print(f">> Pergunta: {pergunta}\n")

    # Forma 1 — cadeia LCEL pura (string entra, string sai)
    cadeia = construir_cadeia_rag(k=4)
    resposta_via_cadeia = cadeia.invoke(pergunta)

    print("=== Resposta via cadeia LCEL ===")
    print(resposta_via_cadeia)
    print()

    # Forma 2 — versão que também devolve as fontes (útil pra UI)
    print("=== Resposta + fontes (versão didática) ===")
    resultado = gerar_resposta(pergunta, k=4)
    print(resultado["resposta"])
    print("\nFontes usadas:")
    for i, f in enumerate(resultado["fontes"], 1):
        print(f"  [{i}] {f['fonte']} p.{f['pagina']} — distância={f['distancia']:.4f}")


if __name__ == "__main__":
    main()
