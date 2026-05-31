"""
Etapa 3 do Agente — Observabilidade

Duas camadas, complementares:

1. **LangSmith (recomendado em produção)**
   Todas as chamadas de LLM, Runnable e nodes do LangGraph aparecem
   automaticamente no painel em https://smith.langchain.com.
   Basta exportar:

       LANGSMITH_TRACING=true
       LANGSMITH_API_KEY=ls__...
       LANGSMITH_PROJECT=ai-lab

   (já fizemos isso no .env da raiz).

2. **Logger JSONL local (didático)**
   Os nodes do nosso grafo gravam eventos em `logs/agente.jsonl`.
   Útil para inspecionar offline com `jq`, pandas ou enviar para
   um Grafana/ELK.

Aqui mostramos como inspecionar o log local e como verificar se o
LangSmith está ativo.
"""

from __future__ import annotations

import os

from agente_lib import ARQUIVO_LOG, logger_jsonl


def status_langsmith() -> None:
    tracing = os.getenv("LANGSMITH_TRACING", "").lower() in {"true", "1", "yes"}
    projeto = os.getenv("LANGSMITH_PROJECT", "default")
    chave = os.getenv("LANGSMITH_API_KEY", "")
    if tracing and chave.startswith("ls__"):
        print(f"[LangSmith] ATIVO — projeto '{projeto}'")
        print(f"            painel: https://smith.langchain.com/o/-/projects/p/{projeto}")
    else:
        print("[LangSmith] DESATIVADO — configure .env para ativar tracing remoto")


def gerar_alguns_eventos() -> None:
    logger_jsonl.evento("demo_iniciada", versao_agente="0.2.0")
    logger_jsonl.evento("ferramenta_chamada", nome="rag_buscar", duracao_ms=137)
    logger_jsonl.evento("ferramenta_chamada", nome="calcular", duracao_ms=2)
    logger_jsonl.evento("demo_encerrada")
    print(f"[JSONL] sessao_id desta execução: {logger_jsonl.sessao_id}")
    print(f"[JSONL] arquivo: {ARQUIVO_LOG}")


def ver_ultimos_eventos(n: int = 5) -> None:
    if not ARQUIVO_LOG.exists():
        print("(arquivo de log ainda não existe — rode primeiro o agente)")
        return
    linhas = ARQUIVO_LOG.read_text(encoding="utf-8").splitlines()
    print(f"\nÚltimas {min(n, len(linhas))} linhas do log:")
    for linha in linhas[-n:]:
        print(" ", linha)


def main() -> None:
    status_langsmith()
    print()
    gerar_alguns_eventos()
    ver_ultimos_eventos(n=5)


if __name__ == "__main__":
    main()
