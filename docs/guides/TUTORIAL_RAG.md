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
3. **[Chunks e Overlaps](#fundamentação-chunks-e-overlaps)** - **ESSENCIAL: Como textos são processados**
4. **[Guardrails em Sistemas RAG](#guardrails-em-sistemas-rag)** - **🛡️ PROTEÇÕES: Segurança e qualidade**
5. **[Componentes Técnicos](#componentes-técnicos)** - Tecnologias utilizadas

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

## 🔪 **FUNDAMENTAÇÃO: CHUNKS E OVERLAPS**

### 🎯 **O que são Chunks?**

**Chunks** são fragmentos de texto de tamanho limitado criados pela divisão de documentos maiores. É o processo mais crítico em sistemas RAG, pois determina:

- **Granularidade**: Quão específica é a informação recuperada
- **Contexto**: Quantidade de informação disponível por busca
- **Performance**: Velocidade de busca e geração de embeddings
- **Qualidade**: Precisão das respostas geradas

### 📐 **Por que Precisamos de Chunks?**

**1. Limitações Técnicas:**
```python
# Limitações típicas de modelos
MAX_TOKENS_EMBEDDING = 8192    # nomic-embed-text
MAX_TOKENS_LLM = 4096          # contexto típico llama3
MAX_CHUNK_SIZE = 500           # nosso padrão
```

**2. Limitações Cognitivas:**
- LLMs têm dificuldade com textos muito longos
- Informação relevante pode "se perder" em meio a muito contexto
- Embeddings de textos grandes são menos precisos

**3. Eficiência Computacional:**
- Busca vetorial é mais rápida com chunks menores
- Menos dados transferidos pela rede
- Processamento paralelo mais eficiente

### 🧩 **Como Funciona o Chunking?**

**Estratégia Hierárquica (RecursiveCharacterTextSplitter):**

```python
# Separadores em ordem de prioridade
separators = [
    "\n\n",    # 1º: Parágrafos (preserva estrutura semântica)
    "\n",      # 2º: Linhas (mantém unidade textual)  
    " ",       # 3º: Palavras (preserva tokens)
    ""         # 4º: Caracteres (último recurso)
]

# Processo inteligente
1. Tenta dividir por parágrafo
2. Se ainda muito grande → divide por linha
3. Se ainda muito grande → divide por palavra
4. Se ainda muito grande → divide por caractere
```

**Exemplo Prático:**
```
Documento Original (1500 chars):
"A inteligência artificial (IA) é uma tecnologia...
[parágrafo 1: 600 chars]

Machine Learning é um subcampo da IA que permite...
[parágrafo 2: 500 chars]

Deep Learning, por sua vez, utiliza redes neurais...
[parágrafo 3: 400 chars]"

Resultado do Chunking (CHUNK_SIZE=500):
┌─────────────────────────────────────┐
│ CHUNK 1 (500 chars)                │
│ "A inteligência artificial (IA)...  │
│ [parágrafo 1 completo]              │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ CHUNK 2 (500 chars)                │  
│ "Machine Learning é um subcampo...  │
│ [parágrafo 2 + início do 3]        │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ CHUNK 3 (320 chars)                │
│ "Deep Learning, por sua vez...      │  
│ [resto do parágrafo 3]              │
└─────────────────────────────────────┘
```

### 🔗 **O que são Overlaps?**

**Overlap** é a sobreposição intencional entre chunks consecutivos. Em nosso sistema: **CHUNK_OVERLAP = 80 caracteres (~16% do chunk)**.

**Visualização do Overlap:**
```
CHUNK 1: "...conceitos de machine learning são fundamentais para..."
                                        ↕ OVERLAP (80 chars)
CHUNK 2: "...são fundamentais para compreender deep learning..."
```

### 🎯 **Por que Overlaps são Necessários?**

**1. Preservação de Contexto:**
```
❌ SEM OVERLAP:
Chunk 1: "A técnica de gradient descent é utilizada para otimizar"
Chunk 2: "funções de custo em redes neurais."
❌ Informação fragmentada!

✅ COM OVERLAP:
Chunk 1: "A técnica de gradient descent é utilizada para otimizar"
Chunk 2: "otimizar funções de custo em redes neurais profundas."
✅ Contexto preservado!
```

**2. Evitar Quebra de Conceitos:**
- Sentenças não ficam cortadas pela metade
- Conceitos relacionados permanecem juntos
- Referências pronominais mantêm sentido

**3. Redundância Estratégica:**
- Múltiplas oportunidades de recuperar informação importante
- Maior chance de encontrar contexto relevante
- Compensação para imprecisões na busca vetorial

### 🗄️ **Como o Banco Vetorial Opera com Chunks?**

**1. Processo de Armazenamento:**
```python
# Para cada chunk individual
chunk_text = "Machine learning é um subcampo..."

# 1. Geração de Embedding
embedding = embedding_model.embed(chunk_text)
# Resultado: vetor de 768 dimensões [0.123, -0.456, 0.789, ...]

# 2. Armazenamento no ChromaDB
vector_store.add(
    documents=[chunk_text],
    embeddings=[embedding], 
    metadatas=[{"source": "ai_tutorial.txt", "chunk_id": 1}],
    ids=["chunk_1"]
)
```

**2. Estrutura de Dados Resultante:**
```
ChromaDB Collection "demo-rag":
├── chunk_1: [embedding_768d] + metadata + texto
├── chunk_2: [embedding_768d] + metadata + texto  
├── chunk_3: [embedding_768d] + metadata + texto
└── ...
```

**3. Processo de Busca:**
```python
# Consulta: "Como funciona machine learning?"

# 1. Embedding da consulta
query_embedding = embedding_model.embed("Como funciona machine learning?")

# 2. Busca por similaridade cosseno
results = vector_store.similarity_search_with_score(
    query_embedding, 
    k=4  # TOP-4 chunks mais similares
)

# 3. Resultado: chunks rankeados por relevância
# Score 0.92: "Machine learning é um subcampo..."
# Score 0.87: "Algoritmos de ML aprendem padrões..."  
# Score 0.82: "Tipos de aprendizado incluem..."
# Score 0.78: "Aplicações práticas de ML..."
```

### 🧮 **Como Funcionam os Embeddings neste Processo?**

**1. Embeddings de Chunks Individuais:**
```python
# Cada chunk vira um ponto no espaço 768-dimensional
chunk_1 = "IA é uma tecnologia revolucionária..."
embedding_1 = [0.12, -0.34, 0.56, ..., 0.78]  # 768 números

chunk_2 = "Machine learning utiliza algoritmos..."  
embedding_2 = [0.15, -0.31, 0.52, ..., 0.81]  # 768 números

# Chunks similares ficam próximos no espaço vetorial
similarity = cosine_similarity(embedding_1, embedding_2)  # 0.87
```

**2. Vantagens de Embeddings por Chunk:**
- **Precisão**: Cada embedding representa um conceito específico
- **Eficiência**: Busca mais rápida em vetores menores
- **Granularidade**: Pode recuperar informação muito específica

**3. Matemática da Busca Vetorial:**
```python
# Similaridade cosseno entre query e chunk
def cosine_similarity(vec_a, vec_b):
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sqrt(sum(a * a for a in vec_a))
    norm_b = sqrt(sum(b * b for b in vec_b))
    return dot_product / (norm_a * norm_b)

# Valores próximos de 1.0 = muito similares
# Valores próximos de 0.0 = pouco similares  
# Valores negativos = opostos semanticamente
```

### ⚖️ **Desafios e Trade-offs de Chunks/Overlaps**

**1. Tamanho dos Chunks:**

| Tamanho | Vantagens | Desvantagens | Quando Usar |
|---------|-----------|--------------|-------------|
| **Pequeno (200-300)** | • Alta precisão<br>• Busca rápida<br>• Menos ruído | • Pouco contexto<br>• Pode fragmentar conceitos<br>• Mais chunks a gerenciar | • Bases de dados técnicas<br>• FAQ estruturado<br>• Respostas curtas |
| **Médio (500-800)** | • Bom balance<br>• Contexto suficiente<br>• Performance adequada | • Trade-off moderado | • **PADRÃO RECOMENDADO**<br>• Documentação geral<br>• Maioria dos casos |
| **Grande (1000-1500)** | • Muito contexto<br>• Preserva estrutura<br>• Menos fragmentação | • Busca mais lenta<br>• Embeddings menos precisos<br>• Pode incluir ruído | • Textos literários<br>• Análises complexas<br>• Contexto muito importante |

**2. Overlaps:**

| Overlap | Vantagens | Desvantagens | Cenário Ideal |
|---------|-----------|--------------|---------------|
| **0% (sem overlap)** | • Sem redundância<br>• Menos armazenamento<br>• Menos processamento | • ❌ Perde contexto<br>• ❌ Fragmenta conceitos<br>• ❌ Referências quebradas | • **NÃO RECOMENDADO**<br>• Apenas em casos de espaço limitado |
| **10-20% (conservador)** | • Preserva contexto básico<br>• Pouca redundância<br>• Eficiente | • Pode ainda fragmentar<br>• Contexto limitado | • Documentos bem estruturados<br>• Textos com seções claras |
| **20-30% (padrão)** | • ✅ **BOM BALANCE**<br>• ✅ Preserva contexto<br>• ✅ Evita fragmentação | • Redundância moderada<br>• Mais armazenamento | • **RECOMENDADO**<br>• Nosso padrão: 16% |
| **40%+ (alto)** | • Máxima preservação<br>• Redundância alta<br>• Contexto garantido | • ❌ Muito armazenamento<br>• ❌ Busca mais lenta<br>• ❌ Informação duplicada | • Textos muito complexos<br>• Contexto crítico |

### 🎯 **Como Decidir Parâmetros de Chunking?**

**Análise do Tipo de Conteúdo:**
```python
# DOCUMENTAÇÃO TÉCNICA
CHUNK_SIZE = 600        # Contexto suficiente para explicações
CHUNK_OVERLAP = 100     # 16% - preserva referências técnicas

# FAQ / RESPOSTAS CURTAS  
CHUNK_SIZE = 300        # Respostas diretas
CHUNK_OVERLAP = 50      # 16% - mínimo necessário

# TEXTOS ACADÊMICOS
CHUNK_SIZE = 800        # Argumentos complexos precisam de contexto
CHUNK_OVERLAP = 150     # 18% - preserva estrutura argumentativa

# CÓDIGO FONTE (se aplicável)
CHUNK_SIZE = 1000       # Funções/classes completas
CHUNK_OVERLAP = 200     # 20% - preserva dependências
```

**Métricas para Otimizar:**
```python
# 1. Execute análise de chunks
python scripts/analyze_chunks.py --full

# 2. Verifique métricas de qualidade  
python scripts/evaluate_rag.py

# 3. Teste diferentes configurações
export CHUNK_SIZE=800
export CHUNK_OVERLAP=150
python scripts/run_ingest.py

# 4. Compare resultados
python scripts/run_query.py "pergunta teste"
```

### 🔬 **Análise Avançada de Chunks**

**Script de Análise Detalhada:**
```bash
# Ver estatísticas completas de chunking
python scripts/analyze_chunks.py --full

# Análises que você verá:
# • Distribuição de tamanhos
# • Overlaps efetivos encontrados  
# • Chunks por documento
# • Conteúdo exato dos overlaps
# • Identificação de problemas
```

**Métricas Importantes:**
- **Tamanho médio real** vs configurado
- **Percentual de overlap efetivo**
- **Chunks "órfãos"** (muito pequenos)
- **Distribuição por documento**
- **Qualidade da preservação de estrutura**

### 🚀 **Otimização Prática**

**1. Configuração Recomendada (nosso padrão):**
```python
CHUNK_SIZE = 500        # Balance contexto/precisão
CHUNK_OVERLAP = 80      # 16% - preserva contexto
RETRIEVAL_K = 4         # TOP-4 chunks mais relevantes
```

**2. Para Experimentar:**
```bash
# Teste chunks maiores para mais contexto
export CHUNK_SIZE=800
export CHUNK_OVERLAP=120

# Teste mais overlaps para máxima preservação  
export CHUNK_OVERLAP=150

# Teste recuperar mais chunks
export RETRIEVAL_K=6
```

**3. Monitoramento Contínuo:**
- Execute `analyze_chunks.py` após mudanças
- Teste com perguntas reais do seu domínio
- Compare qualidade das respostas
- Monitore tempo de resposta

---

## 🛡️ **GUARDRAILS EM SISTEMAS RAG**

### 🎯 **O que são Guardrails?**

**Guardrails** são mecanismos de controle e validação que garantem que sistemas RAG produzam respostas **seguras, precisas e baseadas exclusivamente no contexto** recuperado, evitando alucinações e respostas inventadas.

### 🏗️ **Arquitetura de Guardrails (4 Camadas)**

```
INPUT → [🔍 Input Guards] → RETRIEVAL → [⚖️ Relevance Guards] → 
GENERATION → [📝 Prompt Guards] → OUTPUT → [✅ Output Guards] → RESPOSTA
```

**1. INPUT GUARDRAILS (Pré-processamento):**
- Validação de queries maliciosas (injection attacks)
- Sanitização e normalização de entrada
- Filtro de queries muito curtas/longas

**2. RETRIEVAL GUARDRAILS (Recuperação):**
- **Threshold de similaridade rigoroso** - filtra contexto irrelevante
- Curto-circuito quando não há contexto adequado
- Validação de qualidade dos chunks recuperados

**3. PROMPT GUARDRAILS (Geração):**
- **Templates rigorosos** que limitam escopo do LLM
- Instruções explícitas: "responda APENAS com base no contexto"
- Formato de resposta padronizado

**4. OUTPUT GUARDRAILS (Pós-processamento):**
- Validação de fidelidade ao contexto
- Detecção de alucinações
- Verificação de citação de fontes

### ⚖️ **Threshold de Similaridade - O Coração dos Guardrails**

**O threshold determina quão "similar" um chunk deve ser para ser considerado relevante:**

```python
# ChromaDB usa distância (menor = mais similar)
THRESHOLD_STRICT = 0.25      # Apenas chunks muito relevantes
THRESHOLD_BALANCED = 0.35    # Balance entre precisão e cobertura  
THRESHOLD_PERMISSIVE = 0.50  # Aceita contexto menos relevante
```

**Exemplo prático:**
```bash
# Query: "Como funciona cache distribuído?"
# Chunk 1: "Redis é usado como cache distribuído..." (score: 0.15) ✅ ACEITO
# Chunk 2: "Implementamos microserviços..."        (score: 0.40) ❌ REJEITADO
# Chunk 3: "Cache distribuído com Infinispan..."   (score: 0.18) ✅ ACEITO

# Resultado: Apenas chunks 1 e 3 são enviados ao LLM
```

### 🔧 **Implementação Prática**

**1. Sistema RAG com Guardrails Completos:**
```bash
# Script principal com todas as proteções
python scripts/rag_with_guardrails.py "Qual é a latência das APIs?"

# Diferentes níveis de rigor
python scripts/rag_with_guardrails.py "Como funciona Kubernetes?" strict strict
python scripts/rag_with_guardrails.py "Explique microserviços" balanced balanced
```

**2. Otimização de Threshold:**
```bash
# Análise automática para encontrar threshold ótimo
python scripts/threshold_optimizer.py

# Resultado: recomendação baseada em análise estatística
```

### 📊 **Tipos de Prompt Templates**

**1. STRICT Template (Máxima Segurança):**
```
Você é um assistente que responde EXCLUSIVAMENTE com base no contexto fornecido.

REGRAS OBRIGATÓRIAS:
1. Use APENAS informações presentes no CONTEXTO
2. Se não estiver no contexto: "❌ Não encontrei informações relevantes"
3. NUNCA invente, deduza ou use conhecimento externo
4. Sempre cite a fonte: (Fonte: arquivo.txt)
```

**2. BALANCED Template (Flexibilidade Limitada):**
```
Priorize SEMPRE as informações do CONTEXTO fornecido.
Use conhecimento geral apenas para esclarecimentos básicos.
Indique claramente quando uma informação vem do contexto vs conhecimento geral.
```

### ⚡ **Resultados dos Guardrails**

**Sem Guardrails:**
```
Query: "Como funciona inteligência artificial?"
Resposta: "A inteligência artificial é um campo amplo que inclui machine learning, 
deep learning, processamento de linguagem natural..." [ALUCINAÇÃO - info não está no contexto]
```

**Com Guardrails:**
```
Query: "Como funciona inteligência artificial?"  
Resposta: "❌ Não encontrei informações relevantes no contexto disponível."
[CORRETO - info realmente não está no sistema_completo.txt]
```

**Contexto Encontrado:**
```
Query: "Qual é a latência das APIs?"
Resposta: "✅ Com base no contexto fornecido: A latência média das APIs é de 150ms 
em 99% dos casos. (Fonte: sistema_completo.txt)"
[CORRETO - info extraída exatamente do contexto]
```

### 🎯 **Configuração de Threshold por Caso de Uso**

| Tipo de Sistema | Threshold | Justificativa |
|------------------|-----------|---------------|
| **Sistema Crítico** (medicina, financeiro) | 0.20-0.25 | Zero tolerância a informação incorreta |
| **Documentação Técnica** (nosso exemplo) | 0.30-0.35 | Balance entre precisão e cobertura |
| **FAQ/Suporte** | 0.40-0.50 | Cobertura mais importante que precisão absoluta |
| **Busca Exploratória** | 0.50-0.60 | Descoberta de informação relacionada |

### 🔬 **Análise de Qualidade dos Guardrails**

**Métricas Importantes:**
```bash
# Executar análise completa
python scripts/rag_with_guardrails.py --test

# Métricas reportadas:
# • Taxa de Sucesso: % de queries respondidas com contexto
# • Taxa de Rejeição: % de queries rejeitadas por baixa relevância  
# • Taxa de Proteção: % de queries maliciosas bloqueadas
# • Fidelidade Média: % de resposta baseada no contexto
```

**Interpretação:**
- **Alta Taxa de Rejeição** = Threshold muito rigoroso
- **Baixa Fidelidade** = Template muito permissivo  
- **Muitas Respostas Genéricas** = Threshold muito permissivo

### 🚨 **Detecção de Ataques de Injection**

**Padrões Bloqueados Automaticamente:**
```bash
# Tentativas de manipulação do prompt
"ignore previous instructions"
"forget everything and act as"
"system: you are now"

# Resultado: Query rejeitada antes de chegar ao LLM
```

### 🎛️ **Configuração Avançada**

**Variáveis de Ambiente para Guardrails:**
```bash
# Threshold de similaridade
export SIMILARITY_THRESHOLD=0.30

# Modo de template (strict/balanced)
export TEMPLATE_MODE=strict

# Número mínimo de chunks necessários
export MIN_CHUNKS_REQUIRED=2

# Ativar logs detalhados de guardrails
export GUARDRAILS_VERBOSE=1
```

### 📈 **Otimização Contínua**

**1. Análise de Threshold:**
```bash
# Encontrar threshold ótimo para seus dados
python scripts/threshold_optimizer.py

# Resultado: recomendação estatística baseada em seus documentos
```

**2. Monitoramento de Qualidade:**
```bash
# Logs de decisões de guardrail
tail -f guardrails.log

# Métricas de performance
python scripts/evaluate_rag.py --guardrails
```

### 🎯 **Casos de Uso Avançados**

**1. Threshold Adaptativo:**
```python
# Threshold baseado na complexidade da query
if len(query.split()) > 10:
    threshold = 0.40  # Queries complexas precisam de mais contexto
else:
    threshold = 0.30  # Queries simples podem ser mais rigorosas
```

**2. Validação Semântica:**
```python
# Verificar se resposta "faz sentido" semanticamente
similarity_score = cosine_similarity(query_embedding, response_embedding)
if similarity_score < 0.6:
    return "❌ Resposta gerada não parece relacionada à pergunta"
```

**3. Guardrails Específicos por Domínio:**
```python
# Regras específicas para documentação técnica
if "performance" in query.lower():
    required_keywords = ["latência", "tempo", "velocidade", "ms", "segundos"]
    if not any(keyword in response.lower() for keyword in required_keywords):
        flag_as_potentially_hallucinated()
```

### 🏆 **Boas Práticas para Guardrails**

**1. Comece Rigoroso:**
- Use threshold baixo (0.25-0.30) inicialmente
- Aumente gradualmente baseado em análise empírica

**2. Monitore Continuamente:**
- Log todas as decisões de guardrail
- Analise queries rejeitadas para detectar falsos negativos
- Revise periodicamente threshold baseado em feedback

**3. Teste Adversarial:**
- Teste com queries maliciosas intencionalmente
- Valide que queries fora do domínio são rejeitadas
- Verifique que respostas inventadas são detectadas

**4. Balance Precisão vs Usabilidade:**
- Guardrails muito rigorosos frustram usuários
- Guardrails muito permissivos comprometem qualidade
- Encontre o sweet spot para seu caso de uso

### 🚀 **Implementação em Produção**

**Arquitetura Recomendada:**
```
API Request → Rate Limiting → Input Validation → 
RAG with Guardrails → Output Validation → Response + Metadata
```

**Monitoramento Essencial:**
- Taxa de queries rejeitadas por guardrail
- Tempo médio de processamento por threshold
- Feedback de usuários sobre qualidade das respostas
- Detecção de tentativas de bypass

**Os Guardrails transformam um sistema RAG experimental em um sistema confiável para produção!** 🛡️✨

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

### 6. **analyze_chunks.py** - 🔥 **ANÁLISE ESSENCIAL DE CHUNKING**

**O que faz:**
```bash
python scripts/analyze_chunks.py --full
```

**Análises Críticas:**
- ✅ **Estatísticas de tamanho** dos chunks (média, min, max)
- ✅ **Overlaps entre chunks consecutivos** - detecta e mostra sobreposições
- ✅ **Distribuição por documento** - como cada arquivo foi dividido
- ✅ **Qualidade do chunking** - identifica problemas de fragmentação
- ✅ **Conteúdo exato dos overlaps** - mostra texto sobreposto

**Quando usar:**
- 🎯 **SEMPRE** após configurar ou mudar parâmetros de chunking
- 🔧 Otimizar CHUNK_SIZE e CHUNK_OVERLAP  
- 🐛 Debug de problemas de sobreposição
- 📊 Entender como documentos foram processados
- ⚙️ **Antes de otimizar performance** - veja a seção [Chunks e Overlaps](#fundamentação-chunks-e-overlaps)

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

### 1. **rag_with_guardrails.py** - 🛡️ **RAG COM PROTEÇÕES COMPLETAS**

**O que faz:**
```bash
# RAG básico com guardrails
python scripts/rag_with_guardrails.py "Qual é a latência das APIs?"

# Configurar rigor dos filtros
python scripts/rag_with_guardrails.py "Como funciona Kubernetes?" strict strict
python scripts/rag_with_guardrails.py "Explique microserviços" permissive balanced

# Teste automático com múltiplas queries
python scripts/rag_with_guardrails.py --test
```

**Para que serve:**
- 🛡️ **Proteção contra alucinações** - força aderência ao contexto
- ⚖️ **Filtro de relevância** - threshold de similaridade rigoroso
- 🔒 **Validação de entrada** - detecta tentativas de injection
- ✅ **Validação de saída** - verifica fidelidade e citação de fontes
- 📊 **Métricas de qualidade** - avalia efetividade dos guardrails

**Quando usar:**
- **SEMPRE** em sistemas de produção
- Quando precisar de respostas 100% baseadas no contexto
- Para sistemas críticos (medicina, finanças, legal)
- Análise de segurança e robustez

### 2. **threshold_optimizer.py** - 📊 **OTIMIZAÇÃO AUTOMÁTICA DE THRESHOLD**

**O que faz:**
```bash
# Análise completa e recomendação de threshold ótimo
python scripts/threshold_optimizer.py
```

**Para que serve:**
- 📈 **Análise estatística** de distribuição de scores de similaridade
- 🎯 **Recomendação automática** de threshold baseada em dados
- ⚖️ **Trade-off analysis** entre precisão e cobertura
- 📊 **Visualizações** de performance por threshold
- 🏆 **Threshold ótimo** para seu dataset específico

**Quando usar:**
- **Antes de configurar um sistema RAG** para encontrar parâmetros ideais
- Quando mudar o tipo de documentos/conteúdo
- Para otimização de performance
- Análise científica de qualidade de embeddings

### 3. **evaluate_rag.py** - 🔬 **AVALIAÇÃO DE QUALIDADE**

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

**1. Chunking Strategy (📖 [Ver seção detalhada](#fundamentação-chunks-e-overlaps)):**
```python
# Teste diferentes tamanhos - consulte guia completo acima
CHUNK_SIZES = [250, 500, 1000, 1500]
OVERLAPS = [50, 100, 200]

# Analise resultados com:
python scripts/analyze_chunks.py --full

# Avalie qual combinação funciona melhor para SEU caso
for size in CHUNK_SIZES:
    for overlap in OVERLAPS:
        export CHUNK_SIZE=$size
        export CHUNK_OVERLAP=$overlap
        python scripts/run_ingest.py
        python scripts/run_query.py "sua pergunta teste"
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
