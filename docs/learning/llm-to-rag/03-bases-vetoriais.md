# Capítulo 03 — Bases Vetoriais: Onde a Memória do RAG Vive
### 3.1. Estrutura básica

Uma base vetorial guarda:

- **ID do item**
- **Embedding (vetor)**
- **Metadados** (ex.: arquivo, página, tags)

Exemplo:

```
ID: doc1-chunk3
Vetor: [0.12, -0.43, 0.88, …]
Metadados: {"source": "cache_infinispan.md"}
```o anterior, aprendemos que **embeddings** transformam texto em vetores, permitindo buscar por **significado** em vez de palavras exatas. Mas surge uma questão prática: **onde armazenar esses vetores e como recuperá-los rapidamente quando precisamos?**

A resposta é: **bases vetoriais**.

**Resumo:** neste capítulo explicamos o que são bases vetoriais, como funcionam, por que são essenciais no RAG e como usar o Chroma com exemplos práticos do projeto.

**Sumário:**
1. Por que precisamos de uma base vetorial?
2. O que é uma base vetorial?
3. Como funciona uma base vetorial?
4. Fluxo no RAG com base vetorial
5. Usando o Chroma no seu projeto
6. Exemplo prático
7. Consultando diretamente a base
8. Limitações do Chroma
9. Conclusão

---

## 1. Por que precisamos de uma base vetorial?

Imagine que você tem 50 documentos. Para responder uma pergunta, você poderia:
1. Gerar o embedding da pergunta.
2. Calcular a similaridade com o embedding de cada documento.
3. Escolher os mais próximos.

Simples. Mas e se você tiver **500.000 documentos**? Ou **10 milhões de chunks**?

Fazer comparação de cada novo embedding contra milhões de vetores seria **impraticável**. Precisamos de:
- **Armazenamento eficiente** dos vetores.
- **Busca rápida** dos mais similares.

É isso que uma base vetorial faz.

---

## 2. O que é uma base vetorial?

Uma **base vetorial** é um banco de dados projetado para armazenar vetores e encontrar rapidamente os mais próximos de um vetor de consulta. Ela:
- Armazena cada vetor junto com metadados (ex.: `source`, `id`, `timestamp`).
- Usa estruturas de índice especializadas para busca eficiente.
- Retorna os vetores mais similares a um dado embedding.

Alguns exemplos populares:
- **Chroma** (open source, simples e integrado ao LangChain)
- **FAISS** (Facebook AI)
- **Pinecone** (serviço gerenciado)
- **Milvus** (open source, escalável)

No nosso projeto, usamos **Chroma**, pois ele:
- É local e persistente (SQLite por baixo).
- Integra direto com LangChain.
- É suficiente para demos e produção em pequena/média escala.

---

## 3. Como funciona uma base vetorial?

### 3.1. Estrutura básica
Uma base vetorial guarda:
- **ID do item**
- **Embedding (vetor)**
- **Metadados** (ex.: arquivo, página, tags)

Exemplo:

ID: doc1-chunk3
Vetor: [0.12, -0.43, 0.88, …]
Metadados: {“source”: “cache_infinispan.md”}

### 3.2. Indexação

Para não comparar um vetor com milhões de outros de forma bruta, bases vetoriais usam índices:

- **Flat Index:** compara com todos (bom para poucas entradas).
- **Árvores / Grafos (HNSW, IVF):** aproximam os vizinhos mais próximos, acelerando buscas.

O Chroma, por padrão, é eficiente para milhares/milhões de vetores usando uma camada otimizada com persistência local.

---

## 4. Fluxo no RAG com base vetorial

1. **Ingestão:** documentos são divididos em chunks, embeddings são gerados e salvos na base.
2. **Consulta:** a pergunta vira embedding; a base retorna os `k` chunks mais similares.
3. **Contexto:** esses chunks alimentam o LLM, que gera a resposta.

Sem a base vetorial, teríamos que carregar tudo na memória e recalcular a similaridade manualmente a cada pergunta — inviável.

---

## 5. Usando o Chroma no seu projeto

Já temos no projeto dois arquivos principais:

- `rag_demo/ingest.py`: cria/atualiza a base vetorial com documentos em `data/`.
- `rag_demo/rag.py`: carrega a base, busca embeddings e consulta o LLM.

### 5.1. Estrutura do Chroma
O Chroma cria um diretório (`db/`) contendo:
- Um banco SQLite com os vetores e metadados.
- Arquivos auxiliares para índices.

Isso garante persistência: você injeta documentos uma vez e pode consultá-los depois sem reprocessar tudo.

### 5.2. Ingestão (scripts/run_ingest.py)

Quando você roda:

```bash
python scripts/run_ingest.py
```

O que acontece:

1. Lê arquivos `.txt`, `.md` e `.pdf` em `data/`.
2. Divide em chunks (`RecursiveCharacterTextSplitter`).
3. Gera embeddings (`OllamaEmbeddings`).
4. Salva no Chroma (`Chroma.from_documents(...)`).

### 5.3. Consulta (scripts/run_query.py)

Quando você roda:

```bash
python scripts/run_query.py "O que é paralelismo?"
```

O que acontece:

1. Carrega a base persistida (`Chroma(...)`).
2. Gera embedding da pergunta.
3. Busca os k chunks mais similares (`vectorstore.as_retriever()`).
4. Passa esses chunks para o LLM (`ChatOllama`).
5. Exibe a resposta com contexto.

⸻

6. Exemplo prático

### 6.1. Adicione arquivos a `data/`:

**infinispan_cache.txt:**
```
No Infinispan, a expiração de sessão pode ser configurada via parâmetro 'expiration.max-idle'.
```

**gatos.txt:**
```
Gatos são animais domésticos que dormem em média 16 horas por dia.
```

### 6.2. Ingestão:

```bash
python scripts/run_ingest.py
```

Saída:
```
[INFO] Chunks gerados: 2
[OK] Índice atualizado em 'db/' (coleção: demo-rag)
```

### 6.3. Consulta:

```bash
python scripts/run_query.py "Como configurar expiração de sessão no Infinispan?"
```

Saída:
```
No Infinispan, a expiração de sessão pode ser configurada via 'expiration.max-idle'.
(fonte: infinispan_cache.txt)
```

Note como ele encontrou o documento correto, mesmo sem a pergunta conter exatamente as mesmas palavras do texto.

⸻

7. Consultando diretamente a base

Você pode inspecionar o conteúdo persistido no Chroma com Python:

```python
from langchain_chroma import Chroma

vectorstore = Chroma(
    collection_name="demo-rag",
    persist_directory="db",
)

print("Documentos armazenados:", vectorstore._collection.count())
print("Primeiro item:", vectorstore._collection.get(limit=1))
```

Isso mostra IDs, vetores e metadados. Útil para debug e análise.

⸻

## 8. Limitações do Chroma

- **Persistência local:** bom para demos e produção limitada. Para grande escala, considere FAISS, Milvus ou Pinecone.
- **Reindexação:** precisa reinserir documentos alterados.
- **Indexação aproximada:** não tão avançada quanto soluções enterprise.

Mesmo assim, para aprender e construir RAGs robustos de pequeno a médio porte, Chroma é excelente.

⸻

## 9. Conclusão

Bases vetoriais são a memória do seu RAG. Elas armazenam os embeddings dos seus documentos e permitem buscas rápidas por significado. No seu projeto:

- `scripts/run_ingest.py` constrói essa memória.
- `scripts/run_query.py` a consulta e fornece contexto ao LLM.

No próximo capítulo, vamos ver como construir prompts com contexto recuperado e controlar a forma como o LLM responde — aproximando-nos do RAG completo e confiável.

**Desafio:** adicione 5 documentos sobre diferentes tecnologias em `data/`, injete-os e faça 3 perguntas variadas. Observe quais documentos são recuperados. Depois, inspecione o banco com o snippet de debug acima e veja os vetores armazenados.

---

### Pergunta ao leitor

Agora que entendemos bases vetoriais, quer que eu escreva o **Capítulo 04 — Prompting com Contexto e Controle de Respostas (Guardrails no RAG)** já integrando boas práticas?