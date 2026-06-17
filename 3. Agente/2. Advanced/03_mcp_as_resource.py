"""
MCP como RESOURCE — LLM lendo um "documento" servido pelo MCP

Cenário:
    O servidor `01_mcp_server.py` expõe um RESOURCE em
    `travel://info/international` — uma lista oficial de regras de
    segurança da agência. Aqui o agente NÃO usa nenhuma tool: ele
    apenas LÊ esse texto e injeta diretamente no system prompt do LLM.

Diferença em relação a `02_mcp_as_tool.py`:
    - TOOL    -> o LLM decide CHAMAR uma função remota.
    - RESOURCE-> o cliente BAIXA um conteúdo e injeta como contexto.
    Resource é o jeito "RAG-pelo-MCP": o servidor entrega documentos
    autoritativos (políticas, FAQs, manuais) já versionados num só lugar.

Como rodar:
    python "3. Agente/2. Advanced/03_mcp_as_resource.py"
"""
from __future__ import annotations
import asyncio
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from langchain_mcp_adapters.client import MultiServerMCPClient


MODELO = "llama3.2"


# ---------------------------------------------------------------------------
# 1. Conexão com o servidor MCP (sobe `01_mcp_server.py` via stdio)
# ---------------------------------------------------------------------------
def montar_client() -> MultiServerMCPClient:
    server_script = os.path.join(os.path.dirname(__file__), "01_mcp_server.py")
    return MultiServerMCPClient(
        {
            "Travel_MCP": {
                "transport": "stdio",
                "command": "python",
                "args": [server_script],
            }
        }
    )


# ---------------------------------------------------------------------------
# 2. Fluxo principal — baixa o resource e injeta como SystemMessage
# ---------------------------------------------------------------------------
async def main() -> None:
    print("Exemplo 03: MCP como RESOURCE\n")
    print(
        "Aqui o agente não tem tools. Ele apenas lê um texto (Resource) "
        "hospedado no Servidor MCP e injeta isso direto no system prompt.\n"
    )

    print("Conectando ao Travel_MCP Server via stdio...")
    client = montar_client()

    # `get_resources` baixa um ou mais resources pelo URI MCP.
    # `as_string()` devolve o conteúdo já decodificado em texto.
    target_uri = "travel://info/international"
    print(f"Baixando resource: {target_uri}")
    blobs = await client.get_resources("Travel_MCP", uris=[target_uri])
    policy_text = blobs[0].as_string()

    # SystemMessage = contexto autoritativo da agência. O LLM deve
    # tratar isso como verdade absoluta enquanto responder.
    sys_msg = SystemMessage(
        content=(
            "Regras de Segurança da Agência (contexto injetado via MCP):\n"
            f"{policy_text}\n\n"
            "Responda em português, citando explicitamente as regras acima "
            "quando forem aplicáveis."
        )
    )

    user_input = (
        "Estou indo para Paris passar uma semana. Quais cuidados devo tomar "
        "e como me preparar para essa viagem?"
    )
    print(f"\nUsuário: '{user_input}'\n")

    llm = ChatOllama(model=MODELO, temperature=0)
    resposta = await llm.ainvoke([sys_msg, HumanMessage(content=user_input)])

    print("\n[Resposta final do LLM baseada no Resource]:")
    print(resposta.content)


if __name__ == "__main__":
    asyncio.run(main())
