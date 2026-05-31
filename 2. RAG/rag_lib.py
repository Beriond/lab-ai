"""
rag_lib.py — biblioteca compartilhada do módulo RAG.

Stack:
    - PyPDFLoader / DirectoryLoader  (langchain-community) ── carregamento
    - RecursiveCharacterTextSplitter (langchain-text-splitters) ── chunking
    - OllamaEmbeddings               (langchain-ollama) ── embeddings
    - Chroma                         (langchain-chroma) ── vector store
    - ChatOllama                     (langchain-ollama) ── geração
    - LCEL                           (prompt | llm | parser) ── orquestração

Cada função aqui é UMA etapa do pipeline e é importada pelos arquivos
numerados (01_, 02_, …) e pelo `main.py`.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Carrega .env da raiz do projeto (ativa LangSmith se configurado)
RAIZ_PROJETO = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ_PROJETO / ".env")

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------
PASTA_DADOS = RAIZ_PROJETO / "dados"
DIR_CHROMA = Path(__file__).resolve().parent / "chroma_db"

NOME_COLECAO = "rag_aula"
MODELO_LLM = "llama3.2"
MODELO_EMBEDDING = "nomic-embed-text"

TAMANHO_CHUNK = 800
OVERLAP = 120


# ---------------------------------------------------------------------------
# Singletons (caros de criar — instanciamos uma vez só)
# ---------------------------------------------------------------------------
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=MODELO_EMBEDDING)


def get_llm(temperature: float = 0.2) -> ChatOllama:
    return ChatOllama(model=MODELO_LLM, temperature=temperature)


def get_vector_store() -> Chroma:
    """Retorna o vetor store persistido em disco (cria se não existir)."""
    return Chroma(
        collection_name=NOME_COLECAO,
        embedding_function=get_embeddings(),
        persist_directory=str(DIR_CHROMA),
        collection_metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# 1. Carregamento
# ---------------------------------------------------------------------------
def carregar_pdfs(pasta: Path = PASTA_DADOS) -> list[Document]:
    """
    Lê recursivamente todos os PDFs de `pasta` e retorna uma lista de
    `Document` (1 por página). Cada doc tem `metadata['source']` com o
    caminho do arquivo e `metadata['page']` com o número da página.
    """
    documentos: list[Document] = []
    pdfs = sorted(pasta.rglob("*.pdf"))
    for pdf_path in pdfs:
        loader = PyPDFLoader(str(pdf_path))
        documentos.extend(loader.load())
    return documentos


# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------
def chunkar_documentos(
    documentos: list[Document],
    tamanho: int = TAMANHO_CHUNK,
    overlap: int = OVERLAP,
) -> list[Document]:
    """
    Quebra cada `Document` em chunks menores, preservando os metadados.
    Usa o splitter recursivo (tenta separar primeiro por parágrafo,
    depois linha, depois espaço).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=tamanho,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documentos)


# ---------------------------------------------------------------------------
# 3. Indexação
# ---------------------------------------------------------------------------
def indexar_chunks(chunks: list[Document]) -> Chroma:
    """
    Adiciona os chunks no vector store. `add_documents` gera os
    embeddings via `OllamaEmbeddings` e grava em disco automaticamente.
    """
    vs = get_vector_store()
    if chunks:
        vs.add_documents(chunks)
    return vs


# ---------------------------------------------------------------------------
# 4. Retrieval
# ---------------------------------------------------------------------------
def recuperar(pergunta: str, k: int = 4) -> list[tuple[Document, float]]:
    """Top-k chunks mais próximos da pergunta, com a distância (cosseno)."""
    vs = get_vector_store()
    return vs.similarity_search_with_score(pergunta, k=k)


def get_retriever(k: int = 4):
    """Retorna um Retriever pronto para usar em uma cadeia LCEL."""
    return get_vector_store().as_retriever(search_kwargs={"k": k})


# ---------------------------------------------------------------------------
# 5. Geração (cadeia LCEL)
# ---------------------------------------------------------------------------
PROMPT_RAG = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você é um assistente que responde APENAS com base no contexto fornecido. "
            "Se a resposta não estiver no contexto, diga claramente: "
            "'Não encontrei essa informação nos documentos.' "
            "Cite a fonte (arquivo + página) entre parênteses ao final.",
        ),
        (
            "human",
            'Contexto:\n"""\n{contexto}\n"""\n\nPergunta: {pergunta}',
        ),
    ]
)


def formatar_contexto(docs: list[Document]) -> str:
    """Concatena documentos em um único bloco com cabeçalho de fonte/página."""
    blocos = []
    for d in docs:
        fonte = Path(d.metadata.get("source", "?")).name
        pagina = d.metadata.get("page", "?")
        blocos.append(f"[{fonte} | página {pagina}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocos)


def construir_cadeia_rag(k: int = 4):
    """
    Cadeia LCEL canônica:

        {"pergunta": ...} → recupera docs → formata contexto
                          → prompt → LLM → parser
    """
    retriever = get_retriever(k=k)
    llm = get_llm()

    return (
        {
            "contexto": retriever | formatar_contexto,
            "pergunta": RunnablePassthrough(),
        }
        | PROMPT_RAG
        | llm
        | StrOutputParser()
    )


def gerar_resposta(pergunta: str, k: int = 4) -> dict:
    """
    Versão 'caixa transparente': roda o retrieval explicitamente para
    podermos mostrar as fontes na UI, depois manda pro LLM.
    """
    vs = get_vector_store()
    if vs._collection.count() == 0:  # noqa: SLF001 — só pra UX em aula
        return {
            "resposta": "Base de conhecimento vazia. Rode `python main.py` primeiro.",
            "fontes": [],
        }

    docs_com_score = recuperar(pergunta, k=k)
    docs = [d for d, _ in docs_com_score]
    contexto = formatar_contexto(docs)

    cadeia = PROMPT_RAG | get_llm() | StrOutputParser()
    resposta = cadeia.invoke({"contexto": contexto, "pergunta": pergunta})

    return {
        "resposta": resposta,
        "fontes": [
            {
                "fonte": Path(d.metadata.get("source", "?")).name,
                "pagina": d.metadata.get("page", "?"),
                "distancia": float(score),
            }
            for d, score in docs_com_score
        ],
    }
