"""
utils.py — configurações e singletons compartilhados do módulo RAG.

Stack:
    - PyPDFLoader / DirectoryLoader  (langchain-community) ── carregamento
    - RecursiveCharacterTextSplitter (langchain-text-splitters) ── chunking
    - OllamaEmbeddings               (langchain-ollama) ── embeddings
    - Chroma                         (langchain-chroma) ── vector store
    - ChatOllama                     (langchain-ollama) ── geração
    - LCEL                           (prompt | llm | parser) ── orquestração
"""

from __future__ import annotations
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Carrega .env da raiz do projeto (ativa LangSmith se configurado)
RAIZ_PROJETO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------
PASTA_DADOS = RAIZ_PROJETO / "dados"
DIR_CHROMA = Path(__file__).resolve().parent / "chroma_db"

NOME_COLECAO = "rag_aula"
MODELO_LLM = "llama3.2"
MODELO_EMBEDDING = "nomic-embed-text"

TAMANHO_CHUNK = 500
OVERLAP = 50


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=MODELO_EMBEDDING)


def get_llm(temperature: float = 0.2) -> ChatOllama:
    return ChatOllama(model=MODELO_LLM, temperature=temperature)


def get_vector_store() -> Chroma:
    """Retorna o vetor store persistido em disco (cria se não existir)."""
    return Chroma(
        collection_name=NOME_COLECAO,
        embedding_function=get_embeddings(),
        persist_directory=str(DIR_CHROMA),#pode ser client
        collection_metadata={"hnsw:space": "cosine"},
    )
