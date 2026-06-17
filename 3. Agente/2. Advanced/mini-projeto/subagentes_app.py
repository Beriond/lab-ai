"""
Backend do mini-projeto — Agência de Notícias com Subagentes.

Arquitetura:
                ┌──────────────────────────────────────┐
                │           AGENTE_DIRETOR             │
                │  entende o pedido e delega o tema    │
                └─────┬───────────┬───────────┬────────┘
                      │           │           │
              especialista     especialista  especialista
                esporte         tecnologia   economia
                  ↓                  ↓             ↓
              DuckDuckGo         DuckDuckGo    DuckDuckGo

Cada ESPECIALISTA é um worker (`create_agent`) com uma tool de busca
focada em seu domínio. O DIRETOR decide quais especialistas chamar
baseado no pedido do usuário (pode ser 1, 2 ou os 3).

Cada especialista devolve para o diretor uma lista de notícias com:
    - título
    - link
    - resumo curto

A interface Streamlit (`app.py`) chama `executar` e renderiza apenas
a resposta final do diretor.

Pré-requisitos:
    - Ollama com modelo `llama3.2` baixado
    - `pip install ddgs streamlit`  (já em requirements.txt)
"""
from __future__ import annotations
from typing import Dict, List

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


MODELO_PADRAO = "llama3.2"


# ---------------------------------------------------------------------------
# Wrapper único de busca no DuckDuckGo — cada especialista o usa com um
# "tema" (esporte, tecnologia, economia) que vai concatenado à query.
# Isso evita duplicar a lógica HTTP em 3 lugares.
# ---------------------------------------------------------------------------
def _ddg_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Roda uma busca no DuckDuckGo e devolve [{title, link, body}, ...]."""
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS  # type: ignore

    try:
        bruto = list(DDGS().text(query, max_results=max_results, region="br-pt"))
    except Exception as e:
        return [{"title": "(falha na busca)", "link": "", "body": str(e)}]

    return [
        {
            "title": r.get("title", ""),
            "link": r.get("href") or r.get("url", ""),
            "body": (r.get("body") or "")[:400],
        }
        for r in bruto
    ]


def _formatar_resultados(area: str, query: str, resultados: List[Dict[str, str]]) -> str:
    """Formata a saída de uma tool de busca em string legível para o LLM."""
    if not resultados:
        return f"[{area}] Nenhum resultado encontrado para '{query}'."

    linhas = [f"[{area}] Resultados para '{query}':"]
    for i, r in enumerate(resultados, 1):
        titulo = r.get("title", "(sem título)")
        link = r.get("link", "")
        trecho = r.get("body", "")
        linhas.append(f"{i}. {titulo}\n   {link}\n   {trecho}")
    return "\n".join(linhas)


# ---------------------------------------------------------------------------
# TOOLS DE BAIXO NÍVEL — uma por especialista, todas usam _ddg_search.
# ---------------------------------------------------------------------------
@tool
def buscar_esporte(assunto: str) -> str:
    """Busca notícias recentes de ESPORTE no DuckDuckGo.

    Args:
        assunto: tema ou termo a procurar (ex: "Copa do Mundo 2026",
            "Neymar", "Final Champions").
    """
    query = f"notícias esporte {assunto}"
    return _formatar_resultados("ESPORTE", query, _ddg_search(query))


@tool
def buscar_tecnologia(assunto: str) -> str:
    """Busca notícias recentes de TECNOLOGIA no DuckDuckGo.

    Args:
        assunto: tema (ex: "IA generativa", "lançamento iPhone", "Claude").
    """
    query = f"notícias tecnologia {assunto}"
    return _formatar_resultados("TECNOLOGIA", query, _ddg_search(query))


@tool
def buscar_economia(assunto: str) -> str:
    """Busca notícias recentes de ECONOMIA no DuckDuckGo.

    Args:
        assunto: tema (ex: "dólar hoje", "Selic", "inflação", "Petrobras").
    """
    query = f"notícias economia {assunto}"
    return _formatar_resultados("ECONOMIA", query, _ddg_search(query))


# ---------------------------------------------------------------------------
# CRIANDO OS AGENTES
# ---------------------------------------------------------------------------
def criar_diretor(modelo: str = MODELO_PADRAO, temperatura: float = 0.0):
    """Cria o trio Diretor + 3 Especialistas e devolve o Diretor pronto."""
    llm = ChatOllama(model=modelo, temperature=temperatura)

    # Cada especialista é um agente LLM com UMA tool focada na sua área.
    especialista_esporte = create_agent(
        model=llm,
        tools=[buscar_esporte],
        system_prompt=(
            "Você é um especialista em ESPORTES. Use `buscar_esporte` para "
            "encontrar notícias relevantes e devolva um RESUMO em português "
            "(2-4 frases) seguido dos 3 LINKS mais relevantes."
        ),
    )

    especialista_tecnologia = create_agent(
        model=llm,
        tools=[buscar_tecnologia],
        system_prompt=(
            "Você é um especialista em TECNOLOGIA. Use `buscar_tecnologia` "
            "para encontrar notícias relevantes e devolva um RESUMO em português "
            "(2-4 frases) seguido dos 3 LINKS mais relevantes."
        ),
    )

    especialista_economia = create_agent(
        model=llm,
        tools=[buscar_economia],
        system_prompt=(
            "Você é um especialista em ECONOMIA. Use `buscar_economia` para "
            "encontrar notícias relevantes e devolva um RESUMO em português "
            "(2-4 frases) seguido dos 3 LINKS mais relevantes."
        ),
    )

    # Wrappers — o diretor enxerga cada especialista como uma TOOL.
    @tool
    def acionar_especialista_esporte(assunto: str) -> str:
        """Aciona o especialista em ESPORTES para um assunto.

        Args:
            assunto: tema a pesquisar (ex: "futebol brasileiro").
        """
        r = especialista_esporte.invoke(
            {"messages": [HumanMessage(content=f"Me traga notícias sobre: {assunto}")]}
        )
        return r["messages"][-1].content

    @tool
    def acionar_especialista_tecnologia(assunto: str) -> str:
        """Aciona o especialista em TECNOLOGIA para um assunto.

        Args:
            assunto: tema a pesquisar.
        """
        r = especialista_tecnologia.invoke(
            {"messages": [HumanMessage(content=f"Me traga notícias sobre: {assunto}")]}
        )
        return r["messages"][-1].content

    @tool
    def acionar_especialista_economia(assunto: str) -> str:
        """Aciona o especialista em ECONOMIA para um assunto.

        Args:
            assunto: tema a pesquisar.
        """
        r = especialista_economia.invoke(
            {"messages": [HumanMessage(content=f"Me traga notícias sobre: {assunto}")]}
        )
        return r["messages"][-1].content

    diretor = create_agent(
        model=llm,
        tools=[
            acionar_especialista_esporte,
            acionar_especialista_tecnologia,
            acionar_especialista_economia,
        ],
        system_prompt=(
            "Você é o DIRETOR de uma agência de notícias. Sua função é ENTENDER "
            "o pedido do usuário e DELEGAR ao(s) especialista(s) correto(s).\n"
            "Regras:\n"
            "- Esportes (futebol, NBA, F1, Olimpíadas...) -> `acionar_especialista_esporte`.\n"
            "- Tecnologia (IA, gadgets, software, startups...) -> `acionar_especialista_tecnologia`.\n"
            "- Economia (dólar, juros, bolsa, empresas...) -> `acionar_especialista_economia`.\n"
            "- Se o assunto englobar mais de uma área, chame múltiplos especialistas.\n"
            "- NUNCA invente notícias — sempre acione um especialista.\n"
            "- Ao final, sintetize a resposta em português com seções por área, "
            "mantendo os LINKS que os especialistas devolveram."
        ),
    )
    return diretor


def executar(diretor, pergunta: str) -> str:
    """Roda o diretor e devolve a resposta final em texto."""
    r = diretor.invoke({"messages": [HumanMessage(content=pergunta)]})
    return r["messages"][-1].content
