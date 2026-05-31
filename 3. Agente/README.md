# Módulo 3 — Agente com LangGraph (Memória, Guardrails e Observabilidade)

> **Objetivo da aula:** evoluir o RAG do módulo anterior em um **agente** completo: capaz de manter conversa, decidir quando usar ferramentas, recusar pedidos perigosos e ser totalmente auditável.

---

## 1. O que diferencia um "agente" de um chatbot?

Um **chatbot** responde uma pergunta por vez, sem memória nem capacidade de agir.

Um **agente** tem três superpoderes:

1. **Estado / memória** — lembra do que foi conversado.
2. **Ferramentas (tools)** — pode chamar funções do nosso código (RAG, calculadora, APIs).
3. **Loop de decisão** — observa o resultado da ferramenta e decide o próximo passo.

Para uso real precisamos de mais duas camadas:

4. **Guardrails** — regras determinísticas que filtram entrada e saída.
5. **Observabilidade** — saber exatamente o que o agente fez, com tempo, tokens e custo.

---

## 2. Por que LangGraph?

`langchain` sozinho monta cadeias lineares (`prompt | llm | parser`). Agente exige:

- ramificações ("usar tool?")
- loops ("tool retornou — pensa de novo")
- estado compartilhado entre passos
- persistência por sessão

**LangGraph** modela tudo isso como um **grafo de estado** com nodes e edges. Ele também fornece:

- `ToolNode` — executor pronto que lê `tool_calls` da AIMessage e roda as funções.
- **Checkpointer** — `MemorySaver` (RAM), `SqliteSaver` (disco), `PostgresSaver` (cluster). Memória de curto prazo "de graça", por `thread_id`.
- **Streaming** nativo — `app.stream(...)` mostra cada node sendo executado.
- Integração transparente com **LangSmith**.

---

## 3. Arquitetura do nosso agente

```
                 ┌─────────────────────────────┐
   usuário ───►  │ node guardrail_in           │
                 │ (sanitização, blocklist…)   │
                 └────┬───────────────┬────────┘
              [ok]    │               │ [bloqueado]
                      ▼               ▼
        ┌────────────────────┐       END
        │ node agente        │
        │ ChatOllama         │
        │ .bind_tools([...]) │
        └──┬──────────────┬──┘
   [tool] │               │ [responder]
          ▼               ▼
   ┌──────────┐    ┌──────────────────┐
   │ ToolNode │    │ node guardrail_out│
   └────┬─────┘    │ (PII, gatilhos)   │
        │          └─────────┬─────────┘
        │                    ▼
        └────► (volta p/      END
                agente)

  Toda chamada de node → span no LangSmith
  Memória de curto prazo: MemorySaver (por thread_id)
  Memória de longo prazo: JSON gravado pela tool lembrar_fato
```

---

## 4. Memória

Duas memórias, complementares:

### Curto prazo (sessão)
O estado do grafo (`EstadoAgente`) tem o campo `messages: Annotated[list, add_messages]`. O reducer `add_messages` faz append automático. O **checkpointer** persiste o estado por `thread_id` — você invoca `app.invoke(..., config={"configurable": {"thread_id": "..."}})` e o LangGraph junta com o histórico anterior.

> No nosso `main.py` usamos `MemorySaver` (RAM). Para sobreviver a restart, troque por `SqliteSaver.from_conn_string("...sqlite")`.

### Longo prazo (entre sessões)
Um JSON em `memoria_longo_prazo.json`. A tool `lembrar_fato(chave, valor)` grava. O `system prompt` injeta os fatos a cada turno.

Arquivo: [`01_memoria.py`](./01_memoria.py)

---

## 5. Guardrails

Verificações **determinísticas** (sem LLM) que rodam como nodes do grafo.

| Camada | Node | O que faz |
|--------|------|-----------|
| **Entrada** | `guardrail_in` | bloqueia entrada vazia, gigante, prompt injection, gatilhos proibidos |
| **Saída** | `guardrail_out` | mascara CPF e e-mail, bloqueia conteúdo perigoso |

Por serem nodes, aparecem como spans no LangSmith — você consegue ver exatamente o que foi bloqueado e por quê.

Arquivo: [`02_guardrails.py`](./02_guardrails.py)

---

## 6. Observabilidade

Duas camadas, complementares:

### LangSmith (recomendado em produção)
Basta exportar as envs (já estão no `.env.example`):

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=ls__...
LANGSMITH_PROJECT=ai-lab
```

A partir daí TODA chamada de Runnable e TODO node do grafo aparece no painel, em forma de árvore, com:
- input/output de cada passo
- latência
- tokens (input/output)
- erros

### Logger JSONL local (didático)
Os nodes gravam eventos em `logs/agente.jsonl` — uma linha JSON por evento. Útil para `jq`/pandas e para enviar a um Grafana Loki / ELK depois.

Arquivo: [`03_observabilidade.py`](./03_observabilidade.py)

---

## 7. Ferramentas (Tools)

Cada ferramenta é uma função Python decorada com `@tool`. O decorator extrai nome, descrição (da docstring — **isso é o que o LLM vê** pra escolher) e schema dos parâmetros.

```python
@tool
def rag_buscar(pergunta: str, top_k: int = 3) -> str:
    """Busca trechos relevantes nos PDFs indexados pelo módulo RAG."""
    ...
```

O `ChatOllama` com `llama3.2` suporta tool calling nativo via `.bind_tools([rag_buscar, ...])`. O `ToolNode` do LangGraph executa as chamadas e injeta o resultado de volta no estado como `ToolMessage`.

Ferramentas registradas:

| Tool | Para quê |
|------|----------|
| `rag_buscar(pergunta, top_k)` | Consulta o índice do módulo 2 |
| `lembrar_fato(chave, valor)` | Grava na memória de longo prazo |
| `hora_atual()` | Timestamp ISO |
| `calcular(expressao)` | Calculadora segura (AST) |

Arquivo: [`04_ferramentas.py`](./04_ferramentas.py)

---

## 8. Como rodar

Pré-requisitos:
- Módulo 2 já indexado (`2. RAG/chroma_db/` deve existir).
- Ollama rodando com `llama3.2` e `nomic-embed-text`.

Cada arquivo numerado pode ser rodado isoladamente:

```bash
cd "3. Agente"
python 01_memoria.py
python 02_guardrails.py
python 03_observabilidade.py
python 04_ferramentas.py
```

E para conversar com o agente completo:

```bash
python main.py
```

Comandos especiais dentro do chat:

| Comando | Efeito |
|---------|--------|
| `/sair` | Encerra a sessão |
| `/memoria` | Mostra a memória de longo prazo |
| `/esquecer <chave>` | Apaga uma chave da memória |
| `/historico` | Mostra as mensagens da thread atual |

---

## 9. Vendo o agente no LangSmith

Cada turno do usuário gera uma trace no painel com:

- `LangGraph` (raiz) → `guardrail_in` → `agente` → `tools` → `agente` → `guardrail_out`
- Dentro de `agente`, a chamada do `ChatOllama` com o prompt completo e os `tool_calls` produzidos.
- Dentro de `tools`, uma sub-span por ferramenta executada.

Use isso para depurar prompts, descobrir loops infinitos e medir custo.

---

## 10. Desafios

1. Crie uma tool `clima(cidade)` que retorne um valor mock e veja o agente escolhê-la.
2. Estenda o `guardrail_out` para mascarar telefones brasileiros (`(11) 9XXXX-XXXX`).
3. Substitua o `MemorySaver` por `SqliteSaver` e veja a conversa sobreviver a restart.
4. Adicione um `RunnableConfig` com tags por usuário e filtre no LangSmith.
5. Implemente streaming: `for evento in app.stream({...}, config): print(evento)`.
