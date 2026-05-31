"""
Etapa 1 do Agente — Memória

No LangGraph, há duas memórias:

    - CURTO PRAZO (sessão): o campo `messages` no estado do grafo.
      O LangGraph cuida disso via `add_messages` reducer + checkpointer
      (MemorySaver, SqliteSaver, PostgresSaver…). Cada `thread_id`
      tem seu próprio histórico.

    - LONGO PRAZO (persistente entre sessões): um JSON em disco que o
      agente lê (via system prompt) e escreve (via ferramenta
      `lembrar_fato`).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from agente_lib import (
    ARQUIVO_MEMORIA_LP,
    MemoriaLongoPrazo,
    construir_grafo,
    memoria_lp_global,
)


def demo_memoria_longo_prazo() -> None:
    print("=== Memória de longo prazo (JSON) ===")
    longa = MemoriaLongoPrazo()
    longa.lembrar("nome_usuario", "Maria")
    longa.lembrar("idioma_preferido", "português")
    print(f"Fatos persistidos: {longa.tudo()}")
    print(f"Salvo em: {ARQUIVO_MEMORIA_LP}")


def demo_memoria_curto_prazo() -> None:
    print("\n=== Memória de curto prazo (LangGraph checkpointer) ===")
    print("O grafo guarda o histórico por 'thread_id'. Mesma thread = mesma conversa.\n")

    app = construir_grafo()
    config = {"configurable": {"thread_id": "demo-mem"}}

    # Primeiro turno
    app.invoke(
        {"messages": [HumanMessage(content="Oi, meu nome é Carlos.")]},
        config=config,
    )
    # Segundo turno na MESMA thread
    estado = app.invoke(
        {"messages": [HumanMessage(content="Qual é o meu nome?")]},
        config=config,
    )

    print("Histórico armazenado nessa thread:")
    for m in estado["messages"]:
        tipo = type(m).__name__
        conteudo = (m.content if isinstance(m.content, str) else str(m.content))[:120]
        print(f"  [{tipo}] {conteudo}")


def main() -> None:
    demo_memoria_longo_prazo()
    print(f"\nMemória de longo prazo carregada agora: {memoria_lp_global.tudo()}")
    demo_memoria_curto_prazo()


if __name__ == "__main__":
    main()
