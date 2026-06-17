"""
Subagentes — Agente Vendedor de Viagens

Arquitetura (Supervisor + Workers):

    ┌──────────────────────────────────────────┐
    │            AGENTE VENDEDOR               │
    │  recebe destino + data e delega tarefas  │
    └────────┬─────────────────────┬───────────┘
             │                     │
       guia_turistico          agente_voo
       (Wikipedia)            (DuckDuckGo)

O `vendedor` é um agente LLM cuja única função é DELEGAR.
Cada worker é um `create_agent` próprio, com prompt e ferramentas específicos.
Os workers são expostos ao vendedor como TOOLS (wrapper pattern).

Tools subjacentes (cada uma é "burra" — só faz uma coisa):
    - `wikipedia_summary(consulta)`  -> resumo da Wikipedia em pt-BR
    - `buscar_voos(origem, destino, data)` -> resultados do DuckDuckGo

Pré-requisitos:
    - Ollama rodando com `llama3.2`
    - `pip install ddgs` (já listado em requirements.txt)
    - Acesso à internet (Wikipedia REST API + DuckDuckGo)

Como rodar:
    python "3. Agente/2. Advanced/01_subagents.py"
"""
from __future__ import annotations
import pprint
import requests
from typing_extensions import Annotated

from langchain.agents import create_agent, AgentState
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.types import Command


MODELO = "llama3.2"


# ---------------------------------------------------------------------------
# 1. ESTADO COMPARTILHADO
#    O vendedor preenche `destino` e `data` no início da conversa.
# ---------------------------------------------------------------------------
class VendaState(AgentState):
    origem: str
    destino: str
    data: str


# ---------------------------------------------------------------------------
# 2. TOOLS DE BAIXO NÍVEL — cada worker usa uma
# ---------------------------------------------------------------------------
@tool
def wikipedia_summary(consulta: str) -> str:
    """Busca um resumo da Wikipedia em português para um lugar/tema.

    Args:
        consulta: nome do lugar ou tópico (ex: "Paris", "Torre Eiffel").
    """
    print(f"\n[wikipedia_summary] consulta: {consulta}")
    url = f"https://pt.wikipedia.org/api/rest_v1/page/summary/{consulta.replace(' ', '_')}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return f"Wikipedia não encontrou um artigo para '{consulta}'."
        data = r.json()
        extract = data.get("extract") or f"Artigo encontrado mas sem resumo: {consulta}"
        # Tronca pra não estourar o contexto do LLM com artigos enormes.
        return extract[:1500]
    except Exception as e:
        return f"Falha ao consultar a Wikipedia para '{consulta}': {e}"


@tool
def buscar_voos(origem: str, destino: str, data: str) -> str:
    """Busca opções de voos no DuckDuckGo para a data desejada.

    Se não houver resultado para a data exata, devolve os mais relevantes
    (que costumam ser de datas próximas).

    Args:
        origem: cidade de origem.
        destino: cidade de destino.
        data: data desejada no formato livre (ex: "10/05/2026", "maio 2026").
    """
    print(f"\n[buscar_voos] {origem} -> {destino} em {data}")
    # `ddgs` é o nome novo do antigo `duckduckgo-search`.
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS  

    query = f"voos {origem} para {destino} {data} passagem aérea"
    try:
        resultados = list(DDGS().text(query, max_results=5, region="br-pt"))
    except Exception as e:
        return f"Falha na busca DuckDuckGo: {e}"

    if not resultados:
        return f"Nenhum resultado encontrado para '{query}'."

    linhas = [f"Resultados para '{query}':"]
    for i, r in enumerate(resultados, 1):
        titulo = r.get("title", "(sem título)")
        link = r.get("href") or r.get("url") or ""
        trecho = (r.get("body") or "")[:200]
        linhas.append(f"{i}. {titulo}\n   {link}\n   {trecho}")
    return "\n".join(linhas)


# ---------------------------------------------------------------------------
# 3. SUBAGENTES (workers)
# ---------------------------------------------------------------------------
llm = ChatOllama(model=MODELO, temperature=0)

guia_turistico = create_agent(
    model=llm,
    tools=[wikipedia_summary],
    system_prompt=(
        "Você é um guia turístico. Dado um destino, use a ferramenta "
        "`wikipedia_summary` para descobrir pontos turísticos famosos e "
        "redija uma recomendação CURTA (até 6 frases) em português, "
        "listando 3 a 5 atrações imperdíveis."
    ),
)

agente_voo = create_agent(
    model=llm,
    tools=[buscar_voos],
    system_prompt=(
        "Você é um buscador de voos. Use SEMPRE a ferramenta `buscar_voos` "
        "passando origem, destino e data. Depois, resuma em português as "
        "melhores opções (companhia, links). Se a data exata não retornar "
        "voos, mencione opções próximas que apareceram."
    ),
)


# ---------------------------------------------------------------------------
# 4. TOOLS DO VENDEDOR — wrappers que invocam os workers
# ---------------------------------------------------------------------------
@tool
def consultar_guia_turistico(destino: str) -> str:
    """Aciona o subagente Guia Turístico para um destino.

    Args:
        destino: cidade/local sobre o qual queremos pontos turísticos.
    """
    print(f"\n[vendedor -> guia_turistico] destino={destino}")
    r = guia_turistico.invoke(
        {"messages": [HumanMessage(content=f"Quais pontos turísticos visitar em {destino}?")]}
    )
    return r["messages"][-1].content


@tool
def consultar_agente_voo(origem: str, destino: str, data: str) -> str:
    """Aciona o subagente de Voos para encontrar passagens.

    Args:
        origem: cidade de partida.
        destino: cidade de chegada.
        data: data desejada (formato livre).
    """
    print(f"\n[vendedor -> agente_voo] {origem} -> {destino} em {data}")
    r = agente_voo.invoke(
        {"messages": [HumanMessage(content=f"Ache voos de {origem} para {destino} na data {data}.")]}
    )
    return r["messages"][-1].content


@tool
def registrar_viagem(
    origem: str,
    destino: str,
    data: str | dict,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Registra os dados da viagem (origem, destino, data) no estado.

    Use SOMENTE depois de extrair os três dados do usuário.
    `data` deve ser uma string no formato YYYY-MM-DD.
    """
    # Llama 3.2 às vezes envia a data como dict — formatos vistos:
    # {"value": "..."}, {"date": "..."}, {"data": "..."}, {"type": "date", "value": "..."}.
    # Pegamos o primeiro valor string que parecer uma data.
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, str) and v != "date"), str(data))
    print(f"\n[registrar_viagem] origem={origem}, destino={destino}, data={data}")
    return Command(
        update={
            "origem": origem,
            "destino": destino,
            "data": data,
            "messages": [ToolMessage("Viagem registrada", tool_call_id=tool_call_id)],
        }
    )


# ---------------------------------------------------------------------------
# 5. AGENTE VENDEDOR (supervisor)
# ---------------------------------------------------------------------------
vendedor = create_agent(
    model=llm,
    tools=[registrar_viagem, consultar_guia_turistico, consultar_agente_voo],
    state_schema=VendaState,
    system_prompt=(
        "Você é um VENDEDOR de viagens. Seu trabalho é APENAS coordenar — "
        "nunca invente informações; sempre delegue às tools.\n"
        "Siga exatamente esta ordem:\n"
        "1. Extraia origem, destino e data do usuário e chame `registrar_viagem`.\n"
        "2. Chame `consultar_guia_turistico(destino)` para descobrir pontos turísticos.\n"
        "3. Chame `consultar_agente_voo(origem, destino, data)` para voos.\n"
        "4. Apresente ao cliente um PACOTE em português com os pontos turísticos "
        "   recomendados e as opções de voo encontradas. Seja entusiasmado."
    ),
)


def main() -> None:
    pergunta = (
        "Quero viajar de São Paulo para Paris no dia 10 de julho de 2026. "
        "Me ajude a planejar a viagem."
    )
    print(f"\nCLIENTE: {pergunta}")

    resposta = vendedor.invoke({"messages": [HumanMessage(content=pergunta)]})

    print("\n[VENDEDOR — laudo final]")
    print(resposta["messages"][-1].content)


if __name__ == "__main__":
    main()
