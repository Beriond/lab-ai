"""
Agência de Notícias Multi-Agente — interface Streamlit

Demo de impressionar a sala:
    O usuário digita o que quer saber (ex: "como está a economia hoje
    e o que rolou de IA?"). O AGENTE_DIRETOR interpreta o pedido e
    delega para os especialistas corretos (esporte / tecnologia /
    economia) — cada um faz uma busca real no DuckDuckGo.

    A UI mostra em tempo real:
      - Quais especialistas foram acionados
      - O que cada um devolveu (com LINKS clicáveis)
      - A síntese final do diretor

Como rodar:
    ./.venv/bin/streamlit run "3. Agente/2. Advanced/mini-projeto/app.py"

Pré-requisitos:
    - Ollama rodando localmente com o modelo `llama3.2` baixado.
"""
from __future__ import annotations
import re
import streamlit as st

from subagentes_app import criar_diretor, executar


# ---------------------------------------------------------------------------
# Configuração geral
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Agência de Notícias Multi-Agente",
    page_icon="📰",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cache do diretor — evita recriar agentes a cada interação.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Inicializando agentes (Ollama llama3.2)...")
def get_diretor(modelo: str, temperatura: float):
    return criar_diretor(modelo=modelo, temperatura=temperatura)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuração")
    modelo = st.text_input("Modelo Ollama", value="llama3.2")
    temperatura = st.slider("Temperatura", 0.0, 1.0, 0.0, 0.1)

    st.markdown("---")
    st.markdown("### 🏛️ Arquitetura")
    st.markdown(
        """
        **AGENTE_DIRETOR** (supervisor)
        ↓ delega a tarefa para
        - ⚽ **especialista_esporte**
        - 💻 **especialista_tecnologia**
        - 💰 **especialista_economia**

        Cada um pesquisa no DuckDuckGo e devolve notícias com link
        e resumo. O diretor sintetiza tudo numa única resposta.
        """
    )

    st.markdown("---")
    st.markdown("### 💡 Sugestões de pergunta")
    sugestoes = [
        "Quais são as últimas notícias de IA generativa?",
        "Como está o dólar e a Selic hoje?",
        "Quais as novidades no futebol brasileiro hoje?",
    ]
    for s in sugestoes:
        if st.button(s, use_container_width=True, key=f"sug_{s}"):
            st.session_state["pergunta"] = s


# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------
st.title("📰 Agência de Notícias Multi-Agente")
st.caption(
    "Um **agente diretor** entende seu pedido e aciona **especialistas** "
    "para buscar notícias reais no DuckDuckGo. Powered by Ollama + LangGraph."
)


# ---------------------------------------------------------------------------
# Entrada do usuário
# ---------------------------------------------------------------------------
pergunta = st.text_area(
    "O que você quer saber hoje?",
    value=st.session_state.get("pergunta", ""),
    placeholder="Ex: Como está a economia brasileira e o que rolou de IA esta semana?",
    height=80,
)

executar_clicado = st.button("🚀 Buscar notícias", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Helpers de renderização
# ---------------------------------------------------------------------------
def renderizar_resposta(conteudo: str) -> None:
    """Renderiza a resposta final do diretor, transformando URLs em links clicáveis."""
    url_pattern = re.compile(r"https?://[^\s\)]+")
    for linha in conteudo.split("\n"):
        if url_pattern.search(linha):
            linha_md = url_pattern.sub(lambda m: f"[{m.group(0)}]({m.group(0)})", linha)
            st.markdown(linha_md)
        else:
            st.text(linha)


# ---------------------------------------------------------------------------
# Execução — renderiza apenas a síntese final do diretor.
# ---------------------------------------------------------------------------
if executar_clicado and pergunta.strip():
    diretor = get_diretor(modelo, temperatura)

    with st.spinner("Diretor consultando especialistas..."):
        resposta_final = executar(diretor, pergunta)

    if resposta_final.strip():
        st.markdown("### 📝 Resposta")
        renderizar_resposta(resposta_final)
    else:
        st.warning("O diretor não retornou uma resposta final.")

elif executar_clicado and not pergunta.strip():
    st.warning("Digite uma pergunta antes de executar.")
