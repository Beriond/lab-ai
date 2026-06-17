"""
MCP como TOOL — agente Ollama chamando uma tool exposta por um servidor MCP

Cenário:
    O servidor `01_mcp_server.py` expõe `create_itinerary` como uma TOOL MCP.
    Aqui criamos o CLIENT que:
      1. Sobe o servidor como subprocesso (transporte stdio).
      2. Descobre as tools expostas remotamente via handshake MCP.
      3. Converte essas tools em LangChain tools (via `langchain_mcp_adapters`).
      4. Cria um agente Ollama (llama3.2) com essas tools.
      5. Manda uma pergunta — o agente decide chamar `create_itinerary`.

Por que isso importa?
    Em produção, a mesma API MCP pode ser implementada por outra equipe
    (em outra linguagem!) e o agente continua chamando do mesmo jeito.
    É a maior promessa do protocolo: tools INTEROPERÁVEIS.

Como rodar:
    python "3. Agente/2. Advanced/02_mcp_as_tool.py"
"""
from __future__ import annotations
import asyncio
import os

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

# Adaptador oficial: converte tools MCP em tools LangChain.
from langchain_mcp_adapters.client import MultiServerMCPClient


MODELO = "llama3.2"


# ---------------------------------------------------------------------------
# 1. Conexão com o servidor MCP
#    O MultiServerMCPClient sobe `01_mcp_server.py` como subprocesso e
#    conversa com ele via stdio (stdin/stdout). Em produção poderíamos
#    apontar para um servidor remoto via SSE/HTTP sem mudar o restante.
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
# 2. Fluxo principal — descobre as tools, monta o agente, faz a pergunta
# ---------------------------------------------------------------------------
async def main() -> None:
    print("Exemplo 02: MCP como TOOL (cliente conectado ao Travel_MCP)\n")

    print("Conectando ao Travel_MCP Server via stdio...")
    client = montar_client()

    # `get_tools()` faz handshake MCP, descobre as tools e as embrulha
    # como tools LangChain (compatíveis com `create_agent`).
    langchain_tools = await client.get_tools()
    print(f"[{len(langchain_tools)} tools descobertas]")
    for t in langchain_tools:
        print(f"  - {t.name}")

    print("\nCriando o agente Ollama com as tools MCP...")
    llm = ChatOllama(model=MODELO, temperature=0)
    # O system_prompt é fundamental aqui: modelos pequenos (llama3.2) tendem a
    # confundir o schema do parâmetro com o valor e enviam um dict no lugar de
    # uma string. Deixar explícito que `raw_info` é UMA string única evita o
    # erro `Input should be a valid string` que a tool rejeitaria.
    agente = create_agent(
        llm,
        langchain_tools,
        system_prompt=(
            "Você é um assistente de uma agência de viagens. Para criar "
            "qualquer roteiro, chame SEMPRE a tool `create_itinerary` com "
            "UM ÚNICO argumento chamado `raw_info`, que deve ser uma STRING "
            "em português contendo todas as informações da viagem juntas "
            "(destino, dias, datas, pontos turísticos). Nunca passe um dict "
            "nem múltiplos campos — apenas uma string. Depois, devolva ao "
            "usuário o roteiro."
        ),
    )

    pergunta = (
        "O cliente pediu 7 dias em Paris, visitando a Torre Eiffel, o Arco do "
        "Triunfo, a Catedral de Notre-Dame, o Louvre, o Musée d'Orsay, o "
        "Jardim de Luxemburgo e o Jardim das Tulherias. A viagem será do dia "
        "10 ao dia 17 de Maio. Use a tool create_itinerary para criar o roteiro corporativo."
    )
    print(f"\nUsuário: '{pergunta}'\n")

    # `ainvoke` = versão assíncrona de `invoke`. Necessária aqui porque o
    # MCP client é assíncrono por baixo (precisa falar com o subprocesso).
    resposta = await agente.ainvoke({"messages": [HumanMessage(content=pergunta)]})

    print("\n[Resposta final do agente]:")
    print(resposta["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
