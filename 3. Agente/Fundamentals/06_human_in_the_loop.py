"""
Human-in-the-loop (HITL)

Em casos sensíveis (transferências altas, decisões médicas, exclusão
de dados), queremos que um humano APROVE antes do grafo continuar.

Como funciona no LangGraph:
    1. O nó chama `interrupt(mensagem)` — isso PAUSA a execução e
       devolve o controle para o nosso código.
    2. O estado fica salvo num checkpointer (aqui usamos memória).
    3. Quando o humano decide, chamamos `graph.invoke(Command(resume=...))`
       e o grafo retoma EXATAMENTE de onde parou.

Fluxo:
                          ┌── valor ≤ 5000 → processar_direto → END
    [START] → verificar ──┤
                          └── valor > 5000 → aguardar_aprovacao ⏸  → END
                                                  (pausa para o humano)
"""

from __future__ import annotations
import operator
from typing import Annotated, List, Literal, TypedDict

# Carrega .env da raiz do projeto para ativar o tracing do LangSmith
try:
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

from langgraph.graph import START, END, StateGraph
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver


# Estado da transação
class Estado(TypedDict):
    cliente: str
    valor: float
    status: str
    logs: Annotated[List[str], operator.add]


# 2. Nó de verificação — usa `Command(goto=...)` para escolher dinamicamente
#    o próximo nó (alternativa à aresta condicional da aula anterior).
def verificar(state: Estado) -> Command[Literal["processar_direto", "aguardar_aprovacao"]]:
    print(f"\n[SISTEMA] Verificando R$ {state['valor']:.2f} para {state['cliente']}")

    if state["valor"] > 5000:
        print("[ALERTA] valor alto — encaminhando para aprovação humana")
        return Command(
            update={"logs": [f"aguardando aprovação para R$ {state['valor']:.2f}"]},
            goto="aguardar_aprovacao",
        )

    print("[OK] dentro do limite — processando automaticamente")
    return Command(
        update={"logs": ["processado automaticamente"]},
        goto="processar_direto",
    )


# 3. Nó que PAUSA esperando um humano
def aguardar_aprovacao(state: Estado) -> dict:
    decisao = interrupt(
        f"APROVAR transferência de R$ {state['valor']:.2f} para {state['cliente']}?"
    )
    # A execução só chega aqui DEPOIS que chamarmos Command(resume=...)
    if decisao.lower().startswith("a"):
        print("\n[GERENTE] APROVADO")
        return {"status": "Aprovado", "logs": ["gerente aprovou"]}
    print("\n[GERENTE] REJEITADO")
    return {"status": "Rejeitado", "logs": ["gerente rejeitou"]}


# 4. Nó de processamento direto (sem aprovação)
def processar_direto(state: Estado) -> dict:
    return {"status": "Finalizado", "logs": ["transação concluída"]}


# 5. Construção do grafo
#    O CHECKPOINTER é OBRIGATÓRIO para HITL — sem ele o LangGraph não
#    sabe como retomar a execução após o `interrupt`.
builder = StateGraph(Estado)
builder.add_node("verificar", verificar)
builder.add_node("aguardar_aprovacao", aguardar_aprovacao)
builder.add_node("processar_direto", processar_direto)

builder.add_edge(START, "verificar")
builder.add_edge("aguardar_aprovacao", END)
builder.add_edge("processar_direto", END)

grafo = builder.compile(checkpointer=InMemorySaver())


# 6. Execução interativa
if __name__ == "__main__":
    print("=" * 60)
    print("    SISTEMA BANCÁRIO — Human-in-the-Loop")
    print("=" * 60)

    nome = input("Nome do cliente: ").strip() or "Fulano"
    valor = float(input("Valor da transferência (R$): ") or "0")

    # thread_id identifica essa "conversa" no checkpointer.
    # Mesmo thread_id = mesma transação (mesmo histórico).
    # Tags/metadata: o LangSmith agrupa pelo thread_id e mostra a pausa
    # do interrupt como um nó da árvore — útil pra ver HITL na prática.
    config = {
        "configurable": {"thread_id": "transfer-001"},
        "tags": ["aula:06", "demo:hitl"],
        "metadata": {"aula": "06", "topico": "hitl", "cliente": nome, "valor": valor},
    }

    resultado = grafo.invoke(
        {"cliente": nome, "valor": valor, "status": "pendente", "logs": []},
        config,
    )

    # Se o grafo pausou (interrupt), o resultado terá a chave __interrupt__.
    while "__interrupt__" in resultado:
        msg = resultado["__interrupt__"][-1].value
        print(f"\n>>> PAUSA: {msg}")
        acao = input("Decisão (aprovar/rejeitar): ")
        resultado = grafo.invoke(Command(resume=acao), config)

    print("\n" + "=" * 60)
    print("           STATUS FINAL")
    print("=" * 60)
    print(f"Cliente: {resultado['cliente']}")
    print(f"Valor..: R$ {resultado['valor']:.2f}")
    print(f"Status.: {resultado['status']}")
    print("Logs:")
    for log in resultado["logs"]:
        print(f"  - {log}")
