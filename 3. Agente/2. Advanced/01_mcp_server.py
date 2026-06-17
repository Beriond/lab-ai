"""
Servidor MCP (Model Context Protocol)

O que é MCP?
    MCP é um protocolo aberto (lançado pela Anthropic) que padroniza a forma
    como agentes/LLMs acessam ferramentas, dados e prompts externos. A ideia
    é parecida com a do LSP (Language Server Protocol) para IDEs: em vez de
    cada agente integrar com cada API à mão, ele fala um protocolo único.

Três coisas que um servidor MCP expõe:
    1. TOOLS      — funções que o agente pode chamar (ex: create_itinerary).
    2. RESOURCES  — "documentos" que o agente pode ler (ex: política da empresa).
    3. PROMPTS    — templates de prompt prontos (ex: planejador de roteiro).

Este servidor é a parte SERVER. Ele roda como um subprocesso e fala via
stdio (entrada/saída padrão) com qualquer CLIENT que entenda MCP. Os clients
desta demo estão nos próximos arquivos:
    - `02_mcp_as_tool.py`     — consome as TOOLS do servidor.
    - `03_mcp_as_resource.py` — consome um RESOURCE (texto) do servidor.
    - `04_mcp_as_prompt.py`   — consome um PROMPT template do servidor.

Os scripts client sobem este server automaticamente como subprocesso.
Para rodar isoladamente (apenas para debug):
    python "3. Agente/2. Advanced/01_mcp_server.py"
"""
import sys
from fastmcp import FastMCP

# Nome do servidor — aparece para o client e em logs.
mcp = FastMCP("Travel_MCP")


# --- 1. MCP TOOLS ---------------------------------------------------------
# Tools são funções remotas que o agente pode chamar pelo protocolo.
@mcp.tool()
def create_itinerary(raw_info: str) -> str:
    """Recebe uma descrição livre de viagem e devolve um roteiro estruturado.

    Args:
        raw_info: texto livre do humano (ex: "5 dias em Paris focando em
            história e culinária").
    """
    return f"""
    [ITINERÁRIO ESTRUTURADO GERADO PELO SISTEMA DA AGÊNCIA]
    Contexto Capturado: {raw_info}

    Dia 1: Chegada, check-in no hotel designado, jantar de boas-vindas leve.
    Dia 2: Visita aos pontos turísticos fundamentais pelas redondezas.
    Dia 3: Imersão cultural matutina, tarde livre para compras.
    Dia 4: Experiência gastronômica típica e atividades relaxantes.
    Dia 5: Checkout e partida.
    """


# --- 2. MCP RESOURCES -----------------------------------------------------
# Resources são "URIs" lidas pelo agente — pense em arquivos remotos.
@mcp.resource("travel://info/international")
def get_international_travel_info() -> str:
    """Dicas essenciais e mandatórias de segurança para viagens internacionais."""
    return """
    INFORMAÇÕES OFICIAIS DE SEGURANÇA E VIAGENS INTERNACIONAIS:
    1. Sempre ande com uma cópia colorida do passaporte (deixe o original no cofre).
    2. Tenha atenção a batedores de carteira em capitais europeias turísticas.
    3. Confirme as coberturas do seguro viagem estendido antes de esportes de aventura.
    4. Tenha os contatos da embaixada local no celular.
    """


# --- 3. MCP PROMPTS -------------------------------------------------------
# Prompts são templates parametrizados que o servidor expõe.
@mcp.prompt()
def itinerary_planner(destination: str, days: str, profile: str) -> str:
    """Template oficial de roteiro — junta dados do usuário e políticas internas."""
    policy = get_international_travel_info()
    return f"""Você é o Especialista Chefe de Roteiros Internacionais da Agência.
    Sua missão é criar o discurso de venda perfeito para o destino: {destination}.
    A viagem vai durar {days} dias, perfil de viagem: {profile}.

    Lembre-se SEMPRE de incluir as informações de segurança obrigatórias no final
    da sua fala, que são diretrizes da empresa:
    ===
    {policy}
    ===

    Aja com entusiasmo, gere um roteiro incrível e liste as seguranças com gentileza.
    """


if __name__ == "__main__":
    print("Iniciando Travel_MCP Server (stdio)...")
    try:
        # transport="stdio": o servidor fala com o client via stdin/stdout.
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        print("\nServidor MCP encerrado.")
        sys.exit(0)
