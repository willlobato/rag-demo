# 🚀 Tutorial Completo: RAG com LangChain, ChromaDB e Ollama

> **Sistema Educacional RAG Demo**
> 
> 📚 **[Índice da Documentação](../organization/INDICE_DOCUMENTACAO.md)** | **[Guia de Navegação](GUIA_NAVEGACAO.md)** | **[Classificação de Scripts](../organization/CLASSIFICACAO_SCRIPTS.md)**

Este tutorial fornece uma base teórica sólida para o sistema educacional RAG, preparando você para usar os scripts práticos de forma efetiva.

---

## 📚 **NAVEGAÇÃO RÁPIDA**

### **📖 Fundamentos Teóricos**
1. **[O que é RAG?](#o-que-é-rag)** - Conceitos fundamentais
2. **[Arquitetura do Sistema](#arquitetura-do-sistema)** - Design técnico
3. **[Componentes Técnicos](#componentes-técnicos)** - Tecnologias utilizadas

### **🛠️ Implementação Prática**
4. **[Configuração e Instalação](#configuração-e-instalação)** - Setup inicial
5. **[Scripts Básicos](#scripts-básicos)** - **[Ver Scripts Nível Básico](CLASSIFICACAO_SCRIPTS.md#-nível-básico---fundamentos-essenciais)**
6. **[Scripts Avançados](#scripts-avançados)** - **[Ver Scripts Nível Avançado](CLASSIFICACAO_SCRIPTS.md#-nível-avançado---pesquisa-e-experimentação)**

### **📊 Análise e Otimização**
7. **[Análise e Avaliação](#análise-e-avaliação)** - **[Ver Exemplos Práticos](EXEMPLOS_USO_SCRIPTS.md)**
8. **[Casos de Uso Práticos](#casos-de-uso-práticos)** - **[Ver Documentação Avançada](../reference/DOCUMENTACAO_SCRIPTS_AVANCADOS.md)**
9. **[Otimização e Troubleshooting](#otimização-e-troubleshooting)** - **[Ver Guia de Navegação](GUIA_NAVEGACAO.md#navegação-por-problema-específico)**

### **🚀 Próximos Passos**
10. **[Progressão de Aprendizagem](#próximos-passos)** - **[Ver Fluxo Completo](INDICE_DOCUMENTACAO.md#-fluxo-de-aprendizagem-recomendado)**

---

## 🤖 O que é RAG?

**RAG (Retrieval-Augmented Generation)** é uma técnica de IA que combina:

### 🔍 **Recuperação (Retrieval)**
- Busca informações relevantes em uma base de conhecimento
- Usa **similaridade semântica** (não apenas palavras-chave)
- Trabalha com **embeddings vetoriais**

### ✨ **Geração (Generation)**
- Um LLM (Large Language Model) gera a resposta
- Usa o **contexto recuperado** como base
- Produz respostas **fundamentadas e precisas**

### 🎯 **Por que RAG é Importante?**

| Problema Tradicional | Solução RAG |
|---|---|
| LLMs têm conhecimento limitado | Acesso a dados atualizados |
| Informações podem estar desatualizadas | Base de conhecimento dinâmica |
| Respostas podem ser inventadas ("hallucination") | Respostas baseadas em fontes reais |
| Não cita fontes | Transparência e rastreabilidade |

### 🔄 **Fluxo RAG Simplificado**

```
Pergunta → Busca Vetorial → Contexto → LLM → Resposta + Fontes
```

---

## 🏗️ Arquitetura do Sistema

### 📊 **Diagrama do Fluxo**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documentos    │───▶│   Processamento  │───▶│   ChromaDB      │
│  (TXT/PDF/MD)   │    │   + Chunking     │    │  (Vetores)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                ▲                        │
                                │                        ▼
                        ┌──────────────────┐    ┌─────────────────┐
                        │     TikToken     │    │   Recuperação   │
                        │  (Tokenização)   │    │   Semântica     │
                        └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│   Resposta      │◀───│      LLM         │◀────────────┘
│   + Fontes      │    │   (Llama3)       │
└─────────────────┘    └──────────────────┘
```

### 🔧 **Componentes Principais**

1. **Ingestão**: Carrega e processa documentos
2. **Chunking**: Divide documentos em pedaços menores
3. **Embeddings**: Converte texto em vetores numéricos
4. **Armazenamento**: Salva vetores no ChromaDB
5. **Recuperação**: Busca chunks relevantes
6. **Geração**: LLM produz resposta final

---

## 🛠️ Componentes Técnicos

### 🦜 **LangChain Framework**

**O que é:** Framework para desenvolvimento de aplicações com LLMs

**Por que usar:**
- **Abstração**: Simplifica operações complexas
- **Integrações**: Conecta diferentes ferramentas facilmente
- **Padrões**: Implementa melhores práticas
- **Flexibilidade**: Suporta múltiplos modelos e provedores

**Principais módulos usados:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Chunking
from langchain_chroma import Chroma                                  # Vector Store
from langchain_ollama import OllamaEmbeddings, ChatOllama          # Modelos
from langchain.prompts import ChatPromptTemplate                    # Prompts
from langchain.schema.runnable import RunnablePassthrough           # Chains
```

### 🗄️ **ChromaDB - Vector Database**

**O que é:** Banco de dados especializado em vetores

**Características:**
- **Persistência**: Dados salvos em disco
- **Similaridade**: Busca por cosine similarity
- **Metadados**: Armazena informações adicionais
- **Performance**: Otimizado para embeddings

**Como funciona:**
```python
# Criar/conectar
vect = Chroma(
    collection_name="demo-rag",
    persist_directory="db",
    embedding_function=embeddings
)

# Buscar similares
results = vect.similarity_search_with_score("minha query", k=5)
```

### 🔢 **TikToken - Tokenização**

**O que é:** Biblioteca para contar tokens de modelos OpenAI

**Por que importante:**
- **Limites**: Modelos têm limite de tokens
- **Custos**: APIs cobram por token
- **Chunking**: Garante chunks dentro do limite
- **Otimização**: Controla tamanho de contexto

**Uso prático:**
```python
import tiktoken

# Contar tokens
encoding = tiktoken.get_encoding("cl100k_base")
tokens = encoding.encode("seu texto aqui")
print(f"Tokens: {len(tokens)}")
```

### 🤖 **Ollama - Modelos Locais**

**O que é:** Plataforma para rodar LLMs localmente

**Vantagens:**
- **Privacidade**: Dados não saem da máquina
- **Gratuito**: Sem custos de API
- **Offline**: Funciona sem internet
- **Performance**: Otimizado para hardware local

**Modelos usados:**
- **llama3**: Geração de texto (8B parâmetros)
- **nomic-embed-text**: Embeddings (768 dimensões)

---

## 🎯 Scripts Básicos

### 1. **run_ingest.py** - Indexação de Documentos

**O que faz:**
```bash
python scripts/run_ingest.py
```

**Processo:**
1. Carrega documentos da pasta `data/`
2. Divide em chunks (500 caracteres, overlap 80)
3. Gera embeddings com `nomic-embed-text`
4. Salva no ChromaDB

**Quando usar:**
- Primeira vez que configura o sistema
- Adicionar novos documentos
- Resetar índice: `RESET_CHROMA=1 python scripts/run_ingest.py`

### 2. **run_query.py** - Consultas RAG

**O que faz:**
```bash
python scripts/run_query.py "Como otimizar performance?"
```

**Processo:**
1. Gera embedding da pergunta
2. Busca 4 chunks mais similares
3. Monta prompt com contexto
4. LLM gera resposta
5. Retorna resposta + fontes

**Quando usar:**
- Fazer perguntas sobre seus documentos
- Testar qualidade das respostas
- Demonstrações do sistema

### 3. **list_docs.py** - Explorar Índice

**O que faz:**
```bash
python scripts/list_docs.py
```

**Mostra:**
- Total de chunks indexados
- Conteúdo de cada chunk
- Metadados (fonte, etc.)

**Quando usar:**
- Verificar o que foi indexado
- Debug de problemas de ingestão
- Explorar dados disponíveis

### 4. **search_docs.py** - Busca Semântica

**O que faz:**
```bash
python scripts/search_docs.py "machine learning" 5
```

**Processo:**
- Busca semântica SEM usar LLM
- Retorna chunks + scores de similaridade
- Mais rápido que consulta RAG completa

**Quando usar:**
- Testar qualidade da recuperação
- Análise de relevância
- Debug de embeddings

### 5. **show_vectors.py** - Análise de Vetores

**O que faz:**
```bash
python scripts/show_vectors.py "tecnologia" true
```

**Mostra:**
- Embedding da query (768 dimensões)
- Vetores dos documentos
- Valores numéricos reais
- Normas L2 e estatísticas

**Quando usar:**
- Entender como funcionam embeddings
- Debug de similaridade
- Análise matemática profunda

### 6. **analyze_chunks.py** - Análise de Chunking

**O que faz:**
```bash
python scripts/analyze_chunks.py --full
```

**Análises:**
- Estatísticas de tamanho dos chunks
- Overlaps entre chunks consecutivos
- Distribuição por documento
- Qualidade do chunking

**Quando usar:**
- Otimizar parâmetros de chunking
- Entender como documentos são divididos
- Debug de problemas de sobreposição

### 7. **list_raw.py** - Listagem Simples

**O que faz:**
```bash
python scripts/list_raw.py
```

**Características:**
- Interface mais simples
- Mostra texto completo + embeddings
- Sem parâmetros necessários

**Quando usar:**
- Verificação rápida do índice
- Primeira exploração dos dados
- Interface simples para iniciantes

---

## 🔬 Scripts Avançados

Agora vou implementar scripts avançados para aprofundar seu estudo:

### 1. **evaluate_rag.py** - Avaliação de Qualidade

**Para que serve:**
- Medir qualidade objetiva das respostas
- Comparar diferentes configurações
- Benchmark do sistema
- Métricas científicas

### 2. **analyze_similarity.py** - Análise de Similaridade

**Para que serve:**
- Heatmap de similaridade entre chunks
- Detectar duplicatas
- Clustering de documentos
- Visualização do espaço vetorial

### 3. **experiment.py** - Experimentação

**Para que serve:**
- A/B testing de configurações
- Comparar modelos de embedding
- Testar diferentes chunk sizes
- Batch testing

### 4. **analyze_retrieval.py** - Análise de Recuperação

**Para que serve:**
- Métricas de recuperação (Recall@K, Precision@K)
- Qualidade dos chunks recuperados
- Debug de queries problemáticas
- Análise de ranking

### 5. **advanced_metrics.py** - Métricas Avançadas

**Para que serve:**
- Estatísticas detalhadas dos embeddings
- Distribuições de distância
- Análise de clusters
- Insights matemáticos

---

## 📊 Análise e Avaliação

### Métricas de Qualidade RAG

**1. Métricas de Recuperação:**
- **Recall@K**: % de documentos relevantes recuperados
- **Precision@K**: % de documentos recuperados que são relevantes
- **MRR**: Mean Reciprocal Rank
- **NDCG**: Normalized Discounted Cumulative Gain

**2. Métricas de Geração:**
- **BLEU**: Similaridade com resposta de referência
- **ROUGE**: Overlap de n-gramas
- **Faithfulness**: Fidelidade ao contexto
- **Answer Relevance**: Relevância da resposta

**3. Métricas de Sistema:**
- **Latência**: Tempo de resposta
- **Throughput**: Queries por segundo
- **Consistência**: Variabilidade das respostas

### Ferramentas de Avaliação

**RAGAS Framework:**
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# Avaliar dataset
results = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy]
)
```

---

## 💡 Casos de Uso Práticos

### 1. **Sistema de Documentação Interna**
- Documentos técnicos da empresa
- Manuais e procedimentos
- Base de conhecimento de suporte

### 2. **Assistente de Pesquisa**
- Papers científicos
- Literatura acadêmica
- Relatórios de pesquisa

### 3. **Chatbot de Atendimento**
- FAQ empresarial
- Políticas e regulamentos
- Guias de produto

### 4. **Sistema de Análise Legal**
- Documentos jurídicos
- Precedentes legais
- Contratos e regulamentações

### 5. **Assistente Médico**
- Literatura médica
- Protocolos de tratamento
- Bases de dados clínicas

---

## ⚙️ Otimização e Troubleshooting

### Otimização de Performance

**1. Chunking Strategy:**
```python
# Teste diferentes tamanhos
CHUNK_SIZES = [250, 500, 1000, 1500]
OVERLAPS = [50, 100, 200]

# Avalie qual combinação funciona melhor
for size in CHUNK_SIZES:
    for overlap in OVERLAPS:
        # Execute testes
        pass
```

**2. Embedding Models:**
- `nomic-embed-text`: 768 dim, rápido
- `all-MiniLM-L6-v2`: 384 dim, mais rápido
- `text-embedding-ada-002`: 1536 dim, mais preciso

**3. Retrieval Tuning:**
```python
# Número de chunks recuperados
RETRIEVAL_K = [3, 5, 7, 10]

# Limiar de similaridade
SIMILARITY_THRESHOLD = 0.7
```

### Problemas Comuns

**1. Respostas Irrelevantes:**
- Verificar qualidade dos embeddings
- Ajustar chunk size
- Melhorar prompt template
- Filtrar chunks por score

**2. Performance Lenta:**
- Usar modelo de embedding menor
- Implementar cache de embeddings
- Otimizar índice ChromaDB
- Reduzir número de chunks recuperados

**3. Hallucinations:**
- Ser mais específico no prompt
- Usar temperatura menor
- Implementar validação de resposta
- Adicionar disclaimers

---

## 🚀 Próximos Passos

### Expansões Técnicas

**1. Interface Web:**
```python
# Streamlit
import streamlit as st

st.title("RAG Assistant")
query = st.text_input("Sua pergunta:")
if query:
    response = rag_system.query(query)
    st.write(response)
```

**2. API REST:**
```python
# FastAPI
from fastapi import FastAPI

app = FastAPI()

@app.post("/query")
async def query_rag(question: str):
    return {"answer": rag_system.query(question)}
```

**3. Embeddings Customizados:**
- Fine-tuning de modelos
- Domain-specific embeddings
- Multilingual embeddings

### Funcionalidades Avançadas

**1. Multi-Modal RAG:**
- Imagens + texto
- Tabelas e gráficos
- Documentos estruturados

**2. Conversational RAG:**
- Memória de contexto
- Follow-up questions
- Session management

**3. Hierarchical RAG:**
- Múltiplos níveis de chunking
- Summarização automática
- Routing inteligente

### Ferramentas de Produção

**1. Monitoramento:**
- Logging estruturado
- Métricas de performance
- Alertas de qualidade

**2. Deployment:**
- Containerização (Docker)
- Orquestração (Kubernetes)
- Load balancing

**3. Segurança:**
- Autenticação e autorização
- Rate limiting
- Data privacy

---

## 📚 Recursos de Aprendizado

### Documentação Oficial
- [LangChain](https://python.langchain.com/)
- [ChromaDB](https://docs.trychroma.com/)
- [Ollama](https://ollama.ai/)

### Papers Importantes
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Facebook AI)
- "Dense Passage Retrieval for Open-Domain Question Answering" (Facebook AI)
- "FiD: Fusion-in-Decoder for Open-domain Question Answering" (Facebook AI)

### Cursos e Tutoriais
- LangChain Academy
- DeepLearning.AI RAG Course
- Hugging Face NLP Course

---

## 🎓 Conclusão

Este tutorial cobriu:

✅ **Conceitos fundamentais** de RAG
✅ **Arquitetura técnica** completa
✅ **Implementação prática** com código
✅ **Scripts de análise** e debugging
✅ **Métricas de avaliação**
✅ **Casos de uso** reais
✅ **Otimização** e troubleshooting
✅ **Próximos passos** para expansão

**Próximas ações recomendadas:**
1. Execute todos os scripts básicos
2. Teste com seus próprios documentos
3. Implemente os scripts avançados
4. Experimente diferentes configurações
5. Desenvolva sua própria aplicação

**Lembre-se:** RAG é uma técnica poderosa que combina o melhor de dois mundos - a precisão da busca e a fluência da geração. Com os conhecimentos deste tutorial, você tem tudo que precisa para construir sistemas RAG robustos e eficazes! 🚀
