"""
exercicios_gabarito.py — Respostas comentadas dos 5 desafios

ARQUIVO DESTINADO AO PROFESSOR. Não compartilhe com os alunos antes
da aula; use como referência durante a correção.

Cada desafio aqui traz:
    1. A solução funcional (use `python exercicios_gabarito.py` para rodar).
    2. NOTAS DIDÁTICAS — o que se quer ensinar nesse desafio.
    3. ARMADILHAS COMUNS — erros que os alunos costumam cometer.
    4. EXTENSÕES — variações para discutir em sala se sobrar tempo.

Cada solução é proposital e didaticamente simples. Existe sempre uma
forma mais "elegante" — mas o objetivo aqui é REFORÇAR o conceito do
desafio (few-shot, CoT, role, saída estruturada).
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODELO = "llama3.2"


def _llm(temperature: float = 0.2) -> ChatOllama:
    """Helper para padronizar a criação do cliente. Temperature baixa
    porque QUASE TODOS os desafios querem respostas previsíveis."""
    return ChatOllama(model=MODELO, temperature=temperature)


# ===========================================================================
# Desafio 1 — Resumir em 3 bullets
# ===========================================================================
#
# NOTAS DIDÁTICAS
# ---------------
# O que se quer ensinar: prompt ESPECÍFICO. O aluno precisa explicitar:
#   - quantos bullets (3, exatamente)
#   - tamanho máximo (20 palavras cada)
#   - formato (Markdown com `- `)
#   - língua (português)
#
# Quanto mais restrições, melhor — o modelo respeita razoavelmente bem.
#
# ARMADILHAS COMUNS
# -----------------
# - Aluno pede "resumo curto" → modelo escreve um parágrafo. Sempre QUANTIFICAR.
# - Aluno esquece "Markdown" → vem em texto corrido com travessões.
# - Aluno usa temperature alta → resultados inconsistentes entre runs.
#
# EXTENSÕES
# ---------
# - Trocar o tema do texto e ver se o prompt continua funcionando.
# - Adicionar restrição: "cite uma região por bullet, sem repetir".
# ---------------------------------------------------------------------------
def desafio_1_resumo() -> str:
    texto = (
        "O Brasil tem cinco regiões: Norte, Nordeste, Centro-Oeste, Sudeste e Sul. "
        "Cada região tem características climáticas, culturais e econômicas distintas. "
        "O Sudeste concentra a maior parte do PIB do país, enquanto o Norte abriga "
        "a maior parte da Floresta Amazônica."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Você resume textos em português de forma concisa e estruturada.",
            ),
            (
                "human",
                "Resuma o texto a seguir em EXATAMENTE 3 bullets em Markdown, "
                'cada bullet com no máximo 20 palavras:\n\n"""{texto}"""',
            ),
        ]
    )
    cadeia = prompt | _llm(temperature=0.2) | StrOutputParser()
    return cadeia.invoke({"texto": texto})


# ===========================================================================
# Desafio 2 — Classificação com few-shot
# ===========================================================================
#
# NOTAS DIDÁTICAS
# ---------------
# Reforça o conceito de few-shot: o ALUNO escolheu os labels (SPAM,
# PROMOCAO, IMPORTANTE) — esses NÃO são vocabulário universal. Sem
# exemplos, o modelo pode responder "comercial", "newsletter", "lixo"…
#
# O segredo está em mostrar pelo menos 1 exemplo POR CATEGORIA, no
# formato EXATO que se quer de volta.
#
# ARMADILHAS COMUNS
# -----------------
# - Dar exemplos enviesados (3 de uma categoria, 1 das outras) → o modelo
#   vira papagaio da categoria majoritária.
# - Não usar temperature=0 → resposta varia entre runs.
# - Não pedir "responda APENAS com o rótulo" → vem texto explicativo junto.
#
# EXTENSÕES
# ---------
# - Pedir a resposta como JSON `{"categoria": ..., "confianca": "alta|baixa"}`.
# - Adicionar uma categoria 4ª "DESCONHECIDA" para fallback.
# ---------------------------------------------------------------------------
def desafio_2_classificacao(email: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        "Classifique o e-mail em uma das categorias: SPAM, PROMOCAO, IMPORTANTE.\n"
        "Responda APENAS com a categoria (em maiúsculas).\n\n"
        'E-mail: "Sua conta foi suspensa, clique aqui para reativar imediatamente!"\n'
        "Categoria: SPAM\n\n"
        'E-mail: "Black Friday! Até 70% off em eletrônicos. Ofertas até domingo."\n'
        "Categoria: PROMOCAO\n\n"
        'E-mail: "Reunião com o cliente confirmada para terça, 14h. Pauta em anexo."\n'
        "Categoria: IMPORTANTE\n\n"
        'E-mail: "{email}"\n'
        "Categoria:"
    )
    cadeia = prompt | _llm(temperature=0.0) | StrOutputParser()
    return cadeia.invoke({"email": email}).strip().upper()


# ===========================================================================
# Desafio 3 — Chain-of-Thought
# ===========================================================================
#
# NOTAS DIDÁTICAS
# ---------------
# Enigma clássico — a resposta é "mapa". O modelo costuma chutar
# "fantasma" / "cidade" se for direto. Com CoT, ele "vê" o padrão:
#   - tem características de algo (cidades, montanhas, água)
#   - mas não tem o conteúdo real (casas, árvores, peixes)
#   - logo, é uma REPRESENTAÇÃO de algo, não a coisa em si.
#
# Importante: o prompt CoT precisa pedir explicitamente "pense passo
# a passo" + "termine com 'Resposta final: X'" para a gente extrair.
#
# ARMADILHAS COMUNS
# -----------------
# - Aluno NÃO pede o marcador de resposta final → fica difícil parsear.
# - Aluno usa temperature alta → CoT gera caminhos distintos a cada run.
# - Esquecer que CoT só ajuda em problemas que EXIGEM raciocínio.
#
# EXTENSÕES
# ---------
# - Mostrar 1-3 enigmas resolvidos antes (few-shot CoT, técnica
#   conhecida como "CoT com exemplares").
# - Pedir confidence score: "Quão certo está? 0 a 1".
# ---------------------------------------------------------------------------
def desafio_3_logica() -> str:
    enigma = (
        "Tenho cidades, mas não tenho casas. "
        "Tenho montanhas, mas não tenho árvores. "
        "Tenho água, mas não tenho peixes. O que sou?"
    )
    prompt = ChatPromptTemplate.from_template(
        "{enigma}\n\n"
        "Pense passo a passo:\n"
        "  1. Que tipo de coisa tem cidades sem casas?\n"
        "  2. Que tipo de coisa tem montanhas sem árvores?\n"
        "  3. Que tipo de coisa tem água sem peixes?\n"
        "  4. O que conecta as três pistas?\n\n"
        "Termine com 'Resposta final: <objeto>'."
    )
    cadeia = prompt | _llm(temperature=0.0) | StrOutputParser()
    return cadeia.invoke({"enigma": enigma})


# ===========================================================================
# Desafio 4 — Role prompting
# ===========================================================================
#
# NOTAS DIDÁTICAS
# ---------------
# O ALUNO precisa entender que `SystemMessage` é um "espelho do papel"
# para o modelo — ele orienta tom, vocabulário e profundidade, sem
# precisar repetir contexto em cada pergunta.
#
# A persona pedida ("tech recruiter brasileiro experiente") deve
# carregar:
#   - mercado brasileiro (CLT, PJ, salário em BRL, faixas comuns)
#   - linguagem profissional mas acessível
#   - conselhos pragmáticos, sem clichês
#
# ARMADILHAS COMUNS
# -----------------
# - Persona muito vaga ("você é um recruiter") → respostas genéricas.
# - Esquecer de pedir foco no mercado brasileiro → exemplos em USD,
#   processos americanos, etc.
# - Não definir o tom → pode vir formal demais ou informal demais.
#
# EXTENSÕES
# ---------
# - Encadear num "ChatMessageHistory" para manter conversa em vários
#   turnos com a mesma persona.
# - Acrescentar guardrail: "se a pergunta for ofensiva, recuse educadamente".
# ---------------------------------------------------------------------------
def desafio_4_persona(pergunta: str) -> str:
    persona = (
        "Você é um tech recruiter brasileiro com 15 anos de experiência "
        "recrutando para startups e big techs no Brasil. Conhece bem o "
        "mercado nacional (faixas salariais em BRL, modelos CLT vs PJ, "
        "remoto/híbrido). Seu tom é profissional, direto e empático. "
        "Você dá conselhos pragmáticos, baseados no que costuma funcionar "
        "no mercado brasileiro, e evita clichês motivacionais."
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", persona), ("human", "{pergunta}")]
    )
    cadeia = prompt | _llm(temperature=0.4) | StrOutputParser()
    return cadeia.invoke({"pergunta": pergunta})


# ===========================================================================
# Desafio 5 — Saída estruturada com Pydantic
# ===========================================================================
#
# NOTAS DIDÁTICAS
# ---------------
# A grande sacada é mostrar que `with_structured_output(Schema)` faz
# todo o trabalho que o aluno faria na unha: pedir JSON, parsear,
# validar campos, converter tipos. O retorno é um objeto Pydantic
# pronto para uso em código tipado.
#
# Os `Field(description=...)` são lidos pelo LangChain e VÃO PARA O
# PROMPT — funcionam como documentação para o modelo entender cada
# campo.
#
# ARMADILHAS COMUNS
# -----------------
# - Aluno usa `dict` em vez de Pydantic → perde validação automática.
# - Aluno esquece `description` nos campos ambíguos (preço, tags) → modelo
#   chuta o formato (string "R$ 349,90" em vez de float 349.90).
# - Temperature > 0 → JSON pode vir com pequenas variações entre runs.
#
# EXTENSÕES
# ---------
# - Adicionar campos opcionais (`Optional[str]`) e ver o modelo deixar None.
# - Pedir `List[Produto]` para extrair MÚLTIPLOS produtos de um texto.
# - Usar `Field(ge=0)` para validar preço não-negativo (Pydantic levanta erro).
# ---------------------------------------------------------------------------
class Produto(BaseModel):
    """Produto extraído de uma descrição em texto livre."""

    nome: str = Field(description="Nome curto do produto (sem detalhes técnicos)")
    categoria: str = Field(description="Categoria geral (ex.: eletrodoméstico, vestuário, livro)")
    preco_brl: float = Field(description="Preço em reais. NUMÉRICO. Ex.: 349.90 (não 'R$ 349,90')")
    tags: list[str] = Field(
        default_factory=list,
        description="Lista curta de tags relevantes (cor, voltagem, público-alvo, etc.)",
    )


def desafio_5_estruturado(descricao: str) -> Produto:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Você extrai dados estruturados de descrições de produto em português. "
                "Sempre devolva preço como número decimal (não com 'R$' ou vírgula).",
            ),
            ("human", 'Texto:\n"""{descricao}"""'),
        ]
    )
    llm_estruturado = _llm(temperature=0.0).with_structured_output(Produto)
    cadeia = prompt | llm_estruturado
    return cadeia.invoke({"descricao": descricao})


# ===========================================================================
# Execução demonstrativa (rode `python exercicios_gabarito.py`)
# ===========================================================================
def main() -> None:
    print("═" * 72)
    print(" Desafio 1 — Resumo em 3 bullets")
    print("═" * 72)
    print(desafio_1_resumo())

    print("\n" + "═" * 72)
    print(" Desafio 2 — Classificação com few-shot")
    print("═" * 72)
    casos = [
        "Parabéns! Você ganhou um iPhone GRÁTIS. Clique aqui agora.",
        "Liquidação de inverno! Casacos com 50% off até amanhã.",
        "Anexo segue o contrato revisado. Por favor confirmar até quinta.",
    ]
    for c in casos:
        print(f"  > {c}")
        print(f"    → {desafio_2_classificacao(c)}\n")

    print("═" * 72)
    print(" Desafio 3 — Enigma com Chain-of-Thought")
    print("═" * 72)
    print(desafio_3_logica())

    print("\n" + "═" * 72)
    print(" Desafio 4 — Persona de tech recruiter")
    print("═" * 72)
    perguntas = [
        "Vale a pena pedir aumento na primeira entrevista?",
        "CLT ou PJ: qual escolher quando a empresa oferece os dois?",
    ]
    for p in perguntas:
        print(f"  > {p}")
        print(f"\n{desafio_4_persona(p)}\n")
        print("-" * 72)

    print("\n" + "═" * 72)
    print(" Desafio 5 — Extração estruturada")
    print("═" * 72)
    descricoes = [
        "Cafeteira elétrica preta, 220V, R$ 349,90, ideal para escritório.",
        "Tênis de corrida masculino tamanho 42, cor azul, R$ 489.",
    ]
    for d in descricoes:
        print(f"  > {d}")
        produto = desafio_5_estruturado(d)
        print(f"    {produto.model_dump_json(indent=4)}\n")


if __name__ == "__main__":
    main()
