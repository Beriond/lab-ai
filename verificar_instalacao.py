"""
Script de verificação: roda esse arquivo logo após seguir o README.
Se aparecer [OK] em todas as linhas, sua instalação está pronta.
"""

import os
import sys
from pathlib import Path


def ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def erro(msg: str) -> None:
    print(f"[ERRO] {msg}")


def main() -> int:
    falhas = 0

    # Carrega .env, se existir
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent / ".env")
    except ImportError:
        pass

    # 1. Versão do Python
    if sys.version_info < (3, 10):
        erro(f"Python 3.10+ requerido. Detectado: {sys.version}")
        falhas += 1
    else:
        ok(f"Python {sys.version_info.major}.{sys.version_info.minor} detectado")

    # 2. Ollama: servidor + modelos
    try:
        import ollama

        modelos = ollama.list().get("models", [])
        nomes = {m.get("model", m.get("name", "")) for m in modelos}
        ok("Ollama está rodando")

        if any(n.startswith("llama3.2") for n in nomes):
            ok("Modelo llama3.2 disponível")
        else:
            erro("Modelo llama3.2 não encontrado. Rode: ollama pull llama3.2")
            falhas += 1

        if any("nomic-embed-text" in n for n in nomes):
            ok("Modelo nomic-embed-text disponível")
        else:
            erro("Modelo nomic-embed-text não encontrado. Rode: ollama pull nomic-embed-text")
            falhas += 1
    except ImportError:
        erro("Pacote 'ollama' não instalado. Rode: pip install -r requirements.txt")
        falhas += 1
    except Exception as e:
        erro(f"Não consegui conversar com o Ollama: {e}")
        erro("Garanta que o servidor está rodando: ollama serve")
        falhas += 1

    # 3. langchain core
    try:
        import langchain  # noqa: F401
        import langchain_core  # noqa: F401

        ok("langchain importado")
    except ImportError as e:
        erro(f"langchain não instalado: {e}")
        falhas += 1

    # 4. langchain-ollama (e teste de invoke rápido)
    try:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(model="llama3.2", temperature=0)
        resposta = llm.invoke("Diga apenas a palavra OK.")
        if "OK" in (resposta.content or "").upper():
            ok("langchain-ollama importado e respondendo")
        else:
            ok(f"langchain-ollama respondeu: {resposta.content[:40]!r}")
    except ImportError as e:
        erro(f"langchain-ollama não instalado: {e}")
        falhas += 1
    except Exception as e:
        erro(f"langchain-ollama falhou ao chamar o modelo: {e}")
        falhas += 1

    # 5. langchain-chroma
    try:
        import langchain_chroma  # noqa: F401

        ok("langchain-chroma importado")
    except ImportError as e:
        erro(f"langchain-chroma não instalado: {e}")
        falhas += 1

    # 6. langgraph
    try:
        import langgraph  # noqa: F401

        ok("langgraph importado")
    except ImportError as e:
        erro(f"langgraph não instalado: {e}")
        falhas += 1

    # 7. LangSmith — só reporta o estado (não é obrigatório)
    try:
        import langsmith  # noqa: F401

        tracing = os.getenv("LANGSMITH_TRACING", "").lower() in {"true", "1", "yes"}
        chave = os.getenv("LANGSMITH_API_KEY", "")
        if tracing and chave.startswith("ls__"):
            ok(f"LangSmith tracing ATIVO (projeto: {os.getenv('LANGSMITH_PROJECT', 'default')})")
        else:
            ok("LangSmith DESATIVADO (configure .env se quiser usar)")
    except ImportError as e:
        erro(f"langsmith não instalado: {e}")
        falhas += 1

    print()
    if falhas == 0:
        print("Tudo pronto! Bons estudos.")
        return 0
    print(f"{falhas} verificação(ões) falharam. Releia o README e tente de novo.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
