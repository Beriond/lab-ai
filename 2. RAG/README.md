# Módulo 2 — RAG (LangChain + ChromaDB)

> **Objetivo da aula:** entender por que precisamos de RAG, implementar cada etapa do pipeline canônico usando o stack **LangChain** + **ChromaDB**, e ver a base responder perguntas sobre os seus próprios PDFs.

---

## 1. Por que RAG?

LLMs têm dois problemas conhecidos:

1. **Conhecimento congelado.** Foram treinados até uma data específica. Não sabem o que aconteceu depois nem o que está no PDF interno da sua empresa.
2. **Alucinação.** Quando não sabem, frequentemente *inventam* uma resposta com aparência de verdade.

**RAG = Retrieval-Augmented Generation.** A ideia é simples: antes de pedir a resposta ao LLM, *recuperamos* trechos relevantes da nossa base de conhecimento e *anexamos* esses trechos ao prompt. O LLM responde "com a fonte na mão".

```
Pergunta → [Retriever] → Contexto + Pergunta → [LLM] → Resposta
```

---

## 2. O pipeline canônico (componentes LangChain)

| # | Arquivo | Etapa | Componente LangChain |
|---|---------|-------|----------------------|
| 1 | `01_carregamento.py` | **Load** | `PyPDFLoader` (`langchain-community`) |
| 2 | `02_chunking.py` | **Chunk** | `RecursiveCharacterTextSplitter` |
| 3 | `03_embeddings.py` | **Embed** | `OllamaEmbeddings` (`langchain-ollama`) |
| 4 | `04_indexacao.py` | **Index** | `Chroma.add_documents` (`langchain-chroma`) |
| 5 | `05_retrieval.py` | **Retrieve** | `Chroma.similarity_search_with_score` / `as_retriever()` |
| 6 | `06_geracao.py` | **Generate** | Cadeia **LCEL**: `prompt \| llm \| parser` |

E um orquestrador:

- `main.py` — pipeline ponta a ponta + chat interativo.

---

## 3. Conceitos-chave

### Document
A unidade básica do LangChain. Tem `page_content` (o texto) e `metadata` (dicionário com `source`, `page`, etc.). Todo carregador devolve `Document`, todo splitter consome e devolve `Document`.

### Embedding
Um vetor (768 dimensões para `nomic-embed-text`) que representa o **significado** de um texto. Textos com sentido parecido viram vetores próximos no espaço.

### Chunking
LLMs têm janela de contexto limitada e a recuperação fica mais precisa com pedaços pequenos e coerentes. O `RecursiveCharacterTextSplitter` tenta separar primeiro por parágrafo, depois linha, depois frase. Isso preserva contexto.

- Tamanho típico: 500–1000 caracteres
- Overlap típico: 10–20% do tamanho do chunk

### Vector store
Banco que armazena vetores e responde **busca por similaridade**. O `Chroma` (do `langchain-chroma`) gerencia:
- persistência em disco (`persist_directory=...`)
- embedding automático na hora de adicionar e buscar (`embedding_function=OllamaEmbeddings(...)`)

### Retriever
Abstração do LangChain sobre o vector store. Implementa a interface `Runnable`, ou seja, você pode encadear num pipeline LCEL:

```python
cadeia = {"contexto": retriever | formatar, "pergunta": passthrough} | prompt | llm | parser
```

### LCEL (LangChain Expression Language)
Sintaxe de composição com `|`. Cada peça é um `Runnable`. A cadeia inteira vira um único Runnable com `.invoke`, `.stream`, `.batch`, `.ainvoke` etc.

---

## 4. Pré-requisitos

1. Coloque pelo menos 1 PDF na pasta `dados/`.
2. `venv` ativo e dependências instaladas.
3. Ollama rodando com `llama3.2` e `nomic-embed-text` baixados.
4. (Opcional) LangSmith configurado no `.env` para ver tracing.

---

## 5. Como rodar

Passo a passo (didático):

```bash
cd "2. RAG"
python 01_carregamento.py     # mostra os Documents extraídos dos PDFs
python 02_chunking.py         # mostra os chunks gerados
python 03_embeddings.py       # gera embeddings de exemplo e compara distância
python 04_indexacao.py        # popula o ChromaDB
python 05_retrieval.py        # roda uma busca por similaridade
python 06_geracao.py          # roda a cadeia LCEL completa
```

Pipeline completo + chat:

```bash
python main.py
```

---

## 6. Onde ficam os dados?

- **Fonte:** `dados/*.pdf`
- **Índice gerado:** `2. RAG/chroma_db/` (já está no `.gitignore`)

Para resetar o índice, apague a pasta `chroma_db/` e rode `main.py` de novo.

---

## 7. Vendo o pipeline no LangSmith

Se você configurou `LANGSMITH_TRACING=true` no `.env`, ao rodar qualquer cadeia LCEL você verá no [smith.langchain.com](https://smith.langchain.com) uma árvore de spans com:

- `Retriever.invoke` (com a pergunta e os documentos retornados)
- `ChatPromptTemplate.invoke`
- `ChatOllama.invoke` (com tokens, latência e prompt final)
- `StrOutputParser.invoke`

É a forma mais simples de debugar prompts em produção.

---

## 8. O que tentar depois

- Trocar `chunk_size` e `chunk_overlap` e ver como a qualidade muda.
- Aumentar/diminuir `k` no retriever.
- Usar `MultiQueryRetriever` (gera várias reformulações da pergunta).
- Usar `ContextualCompressionRetriever` para filtrar trechos irrelevantes.
- Trocar `Chroma` por outro vector store (`FAISS`, `Qdrant`, `PGVector`) — só muda a importação.
