# Módulo 1 — Engenharia de Prompt (com LangChain)

> **Objetivo da aula:** entender o que é um LLM, como ele "pensa" e como escrever prompts que produzem respostas úteis, previsíveis e seguras. Tudo usando a abstração `ChatOllama` do **LangChain**.

---

## 1. O que é um LLM?

Um **Large Language Model** (LLM) é uma rede neural treinada para prever o próximo *token* (pedaço de palavra) dado um contexto anterior. Quando você "conversa" com ele, na verdade está pedindo para ele continuar um texto.

Três coisas controlam a saída:

| Elemento | O que é | Analogia |
|----------|---------|----------|
| **Prompt** | O texto que você envia | A pergunta que você faz a um especialista |
| **Contexto** | Tudo que o modelo lê antes de gerar | A documentação que você entrega junto |
| **Parâmetros** | `temperature`, `top_p`, `max_tokens`… | O "humor" do especialista |

### Parâmetros que você vai mexer mais

- **`temperature`** (0.0 a 1.0+): controla aleatoriedade. `0` = determinístico; `1` = criativo.
- **`top_p`**: amostragem por núcleo. Deixe em `0.9` na maioria dos casos.
- **`num_predict`** (no Ollama): número máximo de tokens na resposta.

---

## 2. O modelo de chamadas no LangChain

Em vez de chamar a API do Ollama na unha, usamos o cliente padronizado do LangChain:

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOllama(model="llama3.2", temperature=0.2)
resposta = llm.invoke([
    SystemMessage(content="Você é um assistente conciso."),
    HumanMessage(content="O que é uma API REST?"),
])
print(resposta.content)
```

Por que LangChain?

- **Trocar de provedor é uma linha** (`ChatOllama` → `ChatOpenAI`, `ChatAnthropic`, …).
- **Templates de prompt** versionáveis (`ChatPromptTemplate`).
- **LCEL** (LangChain Expression Language) — compose com `|`:

  ```python
  cadeia = prompt | llm | parser
  cadeia.invoke({"pergunta": "..."})
  ```

- **Tracing automático** no LangSmith.

---

## 3. Anatomia de um bom prompt

Um prompt eficaz costuma ter quatro partes:

1. **Papel (Role):** quem o modelo deve ser? *"Você é um revisor técnico sênior…"*
2. **Tarefa:** o que ele precisa fazer? *"Revise o trecho abaixo e aponte 3 problemas."*
3. **Contexto:** quais dados ele precisa? *"O texto faz parte de um artigo científico em português…"*
4. **Formato de saída:** como você quer a resposta? *"Responda em uma lista numerada em Markdown."*

---

## 4. Técnicas que você vai praticar

| Técnica | Quando usar | Arquivo |
|---------|-------------|---------|
| **Zero-shot** | Tarefa simples, sem exemplo | `02_*` |
| **Few-shot** | Tarefa com formato específico | `02_*` |
| **Chain-of-Thought (CoT)** | Raciocínio em vários passos | `03_*` |
| **Role prompting** | Tom ou expertise específica | `04_*` |
| **Output estruturado** | Quando o resultado vai para outro código | `05_*` |

---

## 5. Princípios de ouro

1. **Seja específico** — *"Resuma em 3 bullets de até 15 palavras cada"* > *"Resuma curto"*.
2. **Mostre, não conte** — exemplos (few-shot) valem mais que adjetivos.
3. **Separe instruções de dados** — use delimitadores (`"""`, `---`, tags XML).
4. **Peça o formato** — se você precisa de JSON, peça JSON e descreva as chaves.
5. **Itere** — prompt é código. Versione, teste, compare.

---

## 6. Como rodar os exercícios

Com o `venv` ativo e o Ollama rodando:

```bash
cd "1. Introducao"
python 01_prompt_basico.py
python 02_zero_shot_vs_few_shot.py
python 03_chain_of_thought.py
python 04_role_prompting.py
python 05_saida_estruturada.py
```

---
