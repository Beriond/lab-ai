"""
Etapa 4 do Agente — Ferramentas (Tools)

No LangChain, uma ferramenta é uma função Python decorada com `@tool`.
O decorator extrai automaticamente:
    - nome (do nome da função)
    - descrição (da docstring) — ISSO É O QUE O LLM LÊ pra decidir usar
    - schema dos parâmetros (das type hints)

O ChatOllama com llama3.2 suporta tool calling nativo: chamamos
`.bind_tools([...])` no LLM e ele passa a retornar `tool_calls`
em `AIMessage`. O `ToolNode` do LangGraph executa essas chamadas.

Ferramentas registradas:
    - rag_buscar     → consulta o índice do módulo 2
    - lembrar_fato   → grava na memória de longo prazo
    - hora_atual     → timestamp ISO
    - calcular       → calculadora segura (AST)
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from agente_lib import FERRAMENTAS, calcular, construir_grafo, hora_atual


def listar_ferramentas() -> None:
    print("Ferramentas disponíveis para o LLM:\n")
    for t in FERRAMENTAS:
        print(f"- {t.name}")
        print(f"    descrição: {t.description.strip().splitlines()[0]}")
        if t.args:
            print(f"    parâmetros: {t.args}")
        print()


def chamar_direto() -> None:
    print("=== Chamando ferramentas diretamente (sem o LLM) ===\n")
    print("calcular.invoke({'expressao': '2 * (3 + 4)'}):")
    print(" →", calcular.invoke({"expressao": "2 * (3 + 4)"}))

    print("\nhora_atual.invoke({}):")
    print(" →", hora_atual.invoke({}))


def usar_via_agente() -> None:
    """Mostra o agente decidindo chamar uma ferramenta sozinho."""
    print("\n=== Agente decidindo qual ferramenta usar ===\n")
    app = construir_grafo()
    config = {"configurable": {"thread_id": "demo-tools"}}

    perguntas = [
        "Quanto é 137 vezes 42?",
        "Que horas são agora?",
    ]
    for p in perguntas:
        print(f">> {p}")
        estado = app.invoke({"messages": [HumanMessage(content=p)]}, config=config)
        print("   →", estado["messages"][-1].content)
        print()


def main() -> None:
    listar_ferramentas()
    chamar_direto()
    usar_via_agente()


if __name__ == "__main__":
    main()
