# AI Lab — Curso Prático de Inteligência Artificial

Bem-vindo(a)! Este repositório é um laboratório prático para você aprender os fundamentos de IA aplicada usando **Python** + **LangChain** + **LangGraph** + **LangSmith** + **ChromaDB**, com modelos rodando 100% locais via **Ollama**.

O curso é dividido em três módulos sequenciais:

1. **Introdução** — Engenharia de Prompt (com `langchain-ollama`)
2. **RAG** — Pipeline canônico (LangChain + ChromaDB)
3. **Agente** — Agente com memória, guardrails e observabilidade (LangGraph + LangSmith)

---

## Stack utilizada

| Camada | Biblioteca | Para quê |
|--------|------------|----------|
| LLM local | **Ollama** (`llama3.2`) | Servidor de modelo na sua máquina |
| Cliente LLM | **`langchain-ollama`** | Wrapper LangChain para Ollama |
| Embeddings | `nomic-embed-text` via `OllamaEmbeddings` | Vetorização semântica |
| Vector store | **`langchain-chroma`** + **`chromadb`** | Busca por similaridade |
| Pipelines | **LangChain** (LCEL) | Compor `prompt → llm → parser` |
| Agentes | **LangGraph** | Grafo de estado com nodes, edges e checkpointing |
| Observabilidade | **LangSmith** | Tracing automático de toda chamada |

---

## Pré-requisitos

### Ferramentas
- **Python 3.10+** instalado
- **Git** instalado
- **Ollama** instalado e rodando (passo 2 abaixo cobre a instalação)

### Modelos do Ollama (baixe antes de começar)

| Modelo | Tamanho | Usado em | Comando |
|--------|---------|----------|---------|
| `llama3.2` | ~2.0 GB | LLM principal (módulos 1, 2 e 3) | `ollama pull llama3.2` |
| `nomic-embed-text` | ~274 MB | Embeddings para o RAG (módulos 2 e 3) | `ollama pull nomic-embed-text` |

> Total: ~2.3 GB de espaço em disco. Reserve mais ~1 GB para `venv` e índice do Chroma.

### Contas externas
- **LangSmith** (opcional, mas recomendado) — conta gratuita em https://smith.langchain.com para tracing do LangChain/LangGraph.

---

## 1. Clonando o repositório

```bash
git clone <URL_DO_SEU_REPO> ai-lab
cd ai-lab
```

---

## 2. Instalando o Ollama e o modelo Llama 3.2

O **Ollama** roda modelos de linguagem (LLMs) localmente, sem precisar de chave de API paga.

### macOS / Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows

Baixe o instalador em https://ollama.com/download e execute.

### Verifique a instalação

```bash
ollama --version
```

### Baixe o modelo Llama 3.2

```bash
ollama pull llama3.2
```

### Baixe também o modelo de embeddings (módulo RAG)

```bash
ollama pull nomic-embed-text
```

### Suba o servidor do Ollama (se ainda não estiver rodando)

```bash
ollama serve
```

> Em macOS o Ollama já inicia como serviço após a instalação. Em Linux/Windows, talvez seja preciso rodar `ollama serve` em outro terminal.

---

## 3. Criando e ativando o ambiente virtual

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Quando o ambiente estiver ativo, você verá `(.venv)` no início do prompt do terminal.

### Instalando as dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Configurando o LangSmith (tracing)

O **LangSmith** é o painel de observabilidade do ecossistema LangChain — ele mostra cada chamada de LLM, cada node do grafo, latência, tokens e erros. Plano gratuito é mais do que suficiente para o curso.

```bash
cp .env.example .env
```

Edite o `.env` e cole sua API key (gerada em https://smith.langchain.com → Settings → API Keys):

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=ls__sua_chave_aqui
LANGSMITH_PROJECT=ai-lab
```

> Se não quiser usar tracing agora, deixe `LANGSMITH_TRACING=false`. O código continua funcionando.

---

## 5. Testando se tudo foi instalado corretamente

Com o `venv` ativo:

```bash
python verificar_instalacao.py
```

Você deve ver algo como:

```
[OK] Python 3.x detectado
[OK] Ollama está rodando
[OK] Modelo llama3.2 disponível
[OK] Modelo nomic-embed-text disponível
[OK] langchain importado
[OK] langchain-ollama importado e respondendo
[OK] langchain-chroma importado
[OK] langgraph importado
[OK] LangSmith tracing ATIVO  (ou DESATIVADO, se você escolheu não usar)
[OK] Tudo pronto! Bons estudos.
```

Se algum item falhar, releia a etapa correspondente acima.

---

## Como percorrer o curso

Siga os módulos na ordem. Cada pasta tem seu próprio `README.md` com a teoria e exercícios:

| Módulo | Pasta | Tema |
|--------|-------|------|
| 1 | [`1. Introducao/`](./1.%20Introducao/) | Engenharia de Prompt com LangChain |
| 2 | [`2. RAG/`](./2.%20RAG/) | Pipeline canônico (LangChain + Chroma) |
| 3 | [`3. Agente/`](./3.%20Agente/) | Agente em LangGraph com memória, guardrails e tracing |

A pasta `dados/` é o local onde você vai colocar os **PDFs** que serão indexados pelo RAG.

---

## Problemas comuns

- **`ollama: command not found`** → reinicie o terminal após instalar.
- **`Connection refused` ao chamar o Ollama** → rode `ollama serve` em outro terminal.
- **`ModuleNotFoundError`** → confirme que o `venv` está ativo e rode `pip install -r requirements.txt` de novo.
- **LangSmith não aparece traços** → confira `LANGSMITH_TRACING=true` no `.env` e se a API key começa com `ls__`.

Bons estudos! 🚀
