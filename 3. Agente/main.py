"""
main.py — agente completo (LangGraph + LangChain + LangSmith).

Arquitetura:

    START → guardrail_in ─[ok]──► agente ─[chamou tool?]──► tools ──┐
              │                     │                                │
              │[bloqueado]          │[respondeu]                     │
              ▼                     ▼                                │
             END               guardrail_out → END ◄─────────────────┘

Cada turno do usuário:
    1. guardrail_in valida a entrada
    2. agente (LLM com tools) decide se responde ou chama tool
    3. tools executa via ToolNode → volta pro agente
    4. guardrail_out sanitiza a resposta antes de devolver

Memória de curto prazo: checkpointer do LangGraph (MemorySaver), por `thread_id`.
Memória de longo prazo: arquivo JSON gravado pela tool `lembrar_fato`.
Observabilidade: LangSmith (tracing automático) + logs JSONL locais.

Comandos especiais no chat:
    /sair        — encerra
    /memoria     — mostra a memória de longo prazo
    /esquecer X  — apaga a chave X da memória
    /historico   — mostra as mensagens da thread atual
"""

from __future__ import annotations

import uuid

from langchain_core.messages import AIMessage, HumanMessage

from agente_lib import construir_grafo, logger_jsonl, memoria_lp_global


def _processar_comando(entrada: str, app, config) -> str | None:
    """Trata comandos `/...`. Retorna a resposta a imprimir, ou None se não for comando."""
    if entrada in ("/sair", "/exit", "/quit"):
        return "__SAIR__"

    if entrada == "/memoria":
        fatos = memoria_lp_global.tudo()
        if not fatos:
            return "(memória de longo prazo vazia)"
        return "\n".join(f"  {k}: {v}" for k, v in fatos.items())

    if entrada.startswith("/esquecer "):
        chave = entrada[len("/esquecer ") :].strip()
        if memoria_lp_global.esquecer(chave):
            return f"Esqueci '{chave}'."
        return f"Não tinha '{chave}' guardado."

    if entrada == "/historico":
        estado = app.get_state(config)
        msgs = estado.values.get("messages", []) if estado else []
        linhas = []
        for m in msgs:
            tipo = type(m).__name__
            conteudo = m.content if isinstance(m.content, str) else str(m.content)
            linhas.append(f"  [{tipo}] {conteudo[:120]}")
        return "\n".join(linhas) if linhas else "(histórico vazio)"

    return None


def main() -> None:
    app = construir_grafo()
    thread_id = uuid.uuid4().hex[:8]
    config = {"configurable": {"thread_id": thread_id}}

    logger_jsonl.evento("sessao_iniciada", thread_id=thread_id)

    print("=" * 60)
    print(" AGENTE LangGraph (memória + guardrails + observabilidade)")
    print(f" thread_id: {thread_id}")
    print(f" sessao_id (logs): {logger_jsonl.sessao_id}")
    print(" comandos: /sair  /memoria  /esquecer <chave>  /historico")
    print("=" * 60)

    while True:
        try:
            entrada = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not entrada:
            continue

        # Comandos locais
        if entrada.startswith("/"):
            saida = _processar_comando(entrada, app, config)
            if saida == "__SAIR__":
                break
            if saida is not None:
                print(saida)
                continue

        # Turno normal: empurra a HumanMessage para o grafo
        try:
            estado_final = app.invoke(
                {"messages": [HumanMessage(content=entrada)]},
                config=config,
            )
        except Exception as e:  # noqa: BLE001
            logger_jsonl.evento("erro", erro=repr(e))
            print(f"\n[ERRO] {e}")
            continue

        # Pega a última AIMessage da conversa
        ultima_ai = next(
            (m for m in reversed(estado_final["messages"]) if isinstance(m, AIMessage)),
            None,
        )
        if ultima_ai is not None:
            conteudo = ultima_ai.content if isinstance(ultima_ai.content, str) else str(ultima_ai.content)
            print(f"\n{conteudo}")

    logger_jsonl.evento("sessao_encerrada")
    print("Até a próxima!")


if __name__ == "__main__":
    main()
