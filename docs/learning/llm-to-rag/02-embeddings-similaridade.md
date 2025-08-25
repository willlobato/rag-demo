# Capítulo 02 — Embeddings e Similaridade: A Ponte Entre Linguagem e Matemática

No capítulo anterior, entendemos que um LLM (Large Language Model) é, no fundo, um previsor de palavras em larga escala, com capacidade de gerar textos coerentes e úteis. Mas vimos também suas limitações: ele não sabe tudo, tem memória limitada e pode "alucinar" respostas.

Como podemos dar a ele acesso a **conhecimento atualizado e específico**? Como podemos fazer isso de forma **eficiente e relevante**, sem enviar um livro inteiro para o modelo a cada pergunta?

A resposta é: **embeddings**.**Resumo:** neste capítulo explicamos o que são embeddings, por que eles importam, como medir similaridade e como usá-los no pipeline RAG, com exemplos práticos.

**Sumário:**
1. Por que embeddings?
2. O que é um embedding?
3. Intuição: significado como posição no espaço
4. Similaridade: como medir proximidade
5. Embeddings no RAG
6. Prática: gerando embeddings e comparando similaridade
7. Experimento com seu projeto
8. Limitações
9. Conclusão

---

## 1. Por que embeddings?

Embeddings transformam texto em **números com significado**. Em vez de buscar por palavras exatas, buscamos por **ideias**. Imagine que você possui 10.000 documentos técnicos e precisa responder rapidamente:
> "Como configuro o cache do Infinispan para expiração de sessão?"

Se você usar busca tradicional, encontrará todos os documentos que contêm as palavras “cache” e “sessão” — incluindo páginas irrelevantes. Mas se alguém escreveu “defina o TTL para armazenar sessões temporárias”, você perderia esse documento, mesmo ele sendo essencial.

Com embeddings, representamos cada trecho (chunk) de cada documento em um espaço matemático onde **significados próximos estão próximos entre si**. Ao buscar “expiração de sessão”, encontraremos também “TTL de sessão” ou “tempo de vida do cache”, mesmo sem as mesmas palavras.

---

## 2. O que é um embedding?

Um **embedding** é um vetor — uma lista de números — que codifica o significado de um texto.

Por exemplo:

```python
"paralelismo"          → [0.12, -0.43, 0.88, …]
"execução simultânea"  → [0.10, -0.40, 0.85, …]
"cachorro"             → [-0.75, 0.31, -0.44, …]
```

Os dois primeiros estão semanticamente próximos, logo seus vetores são semelhantes. O último é distante.

Esses vetores são gerados por modelos treinados em grandes volumes de texto. Modelos modernos, como `nomic-embed-text`, `bge-m3` ou `text-embedding-3-small`, criam embeddings para frases, parágrafos ou documentos inteiros.

---

## 3. Intuição: significado como posição no espaço

Imagine um gráfico 2D:

- Eixo X: tecnologia
- Eixo Y: animais

Se você posicionar:

- “cachorro” ficará em Y alto, X baixo.
- “paralelismo” ficará em X alto, Y baixo.
- “robô-cão” ficará entre os dois.

Em embeddings reais, temos centenas ou milhares de dimensões, mas a lógica é a mesma: **quanto mais próximos dois vetores, mais semelhantes seus significados**.

---

## 4. Similaridade: como medir proximidade?

Precisamos de uma métrica para dizer “esses dois embeddings representam ideias próximas?”.
A mais usada é a **similaridade do cosseno**:

```
similaridade = (A · B) / (||A|| * ||B||)
```

- `A · B`: produto escalar dos vetores
- `||A||`, `||B||`: magnitude (comprimento) dos vetores

O resultado varia entre:

- **1.0:** mesma direção (significados próximos)
- **0.0:** ortogonais (sem relação)
- **-1.0:** opostos

Por que usar o cosseno? Porque ele ignora magnitude e foca na **direção**, que é o que importa para significado.

---

## 5. Embeddings no RAG

No **RAG (Retrieval-Augmented Generation)**:

1. **Ingestão:** dividimos documentos em chunks e geramos embeddings para cada chunk.
2. **Armazenamento:** salvamos esses embeddings em uma base vetorial (como Chroma).
3. **Consulta:** a pergunta do usuário é convertida em embedding; buscamos os chunks mais próximos.
4. **Resposta:** entregamos os chunks relevantes ao LLM, que responde com base neles.

Sem embeddings, o RAG seria impossível.

---

## 6. Prática: gerando embeddings e comparando similaridade

### Setup

Certifique-se de que o Ollama está rodando:

```bash
brew services start ollama
pip install langchain-ollama
```

Código de exemplo:

```python
from langchain_ollama import OllamaEmbeddings
from numpy import dot
from numpy.linalg import norm

embed = OllamaEmbeddings(model="nomic-embed-text")

f1 = "Paralelismo melhora a performance ao rodar tarefas simultâneas."
f2 = "Executar várias tarefas ao mesmo tempo aumenta a eficiência."
f3 = "Gatos dormem muito e são independentes."

e1 = embed.embed_query(f1)
e2 = embed.embed_query(f2)
e3 = embed.embed_query(f3)

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

print("f1 vs f2:", cosine_similarity(e1, e2))
print("f1 vs f3:", cosine_similarity(e1, e3))
```

Saída típica:

```
f1 vs f2: 0.90+
f1 vs f3: 0.05-
```

Resultado: frases do mesmo tema têm alta similaridade; frases sem relação, baixa.

---

## 7. Experimento com seu projeto

Crie dois arquivos em `data/`:

**paralelismo.txt:**
```
Paralelismo permite a execução de múltiplas tarefas em paralelo, reduzindo o tempo total de processamento.
```

**gatos.txt:**
```
Gatos domésticos dormem até 16 horas por dia e têm comportamento independente.
```

Depois rode:

```bash
python scripts/run_ingest.py
python scripts/run_query.py "O que é paralelismo?"
```

O sistema encontrará apenas o conteúdo de `paralelismo.txt`.
Esse é o poder dos embeddings: buscar pelo significado.

---

## 8. Limitações

- Embeddings ruins = recuperação ruim.
- Precisa de índice eficiente para escalar (vamos ver com Chroma).
- Precisa ser atualizado quando novos documentos chegam.

---

## 9. Conclusão

Embeddings são a ponte entre linguagem e matemática. Eles transformam texto em vetores que capturam significado e permitem que LLMs trabalhem com dados externos de forma eficiente. São o coração do RAG: sem eles, não há como conectar seu LLM à sua base de conhecimento.

No próximo capítulo, vamos aprender a armazenar e consultar embeddings em uma base vetorial real com Chroma — e integrá-la ao seu pipeline.

---

### Pergunta ao leitor

Agora que entendemos embeddings, quer que eu escreva o **Capítulo 03 — Bases Vetoriais (Chroma, indexação e recuperação)** com exemplos práticos do seu repositório?