# 📖 Glossário Completo - Conceitos RAG e Scripts Avançados

## 🎯 Termos Fundamentais

### **RAG (Retrieval-Augmented Generation)**
Sistema que combina busca de informações (retrieval) com geração de texto (generation) para criar respostas mais precisas e contextualizadas.

### **Embedding**
Representação vetorial de texto em um espaço n-dimensional, onde palavras/documentos similares ficam próximos no espaço.

### **Vector Store / Vectorstore**
Banco de dados especializado em armazenar e buscar vetores (embeddings) de forma eficiente.

### **Chunk**
Pedaço de texto de tamanho limitado, resultado da divisão de documentos maiores para processamento.

### **Chunking**
Processo de dividir documentos longos em pedaços menores para criar embeddings mais focados.

---

## 📊 Métricas Matemáticas

### **Similaridade Cosseno**
- **Fórmula**: `cos(θ = (A·B) / (||A|| × ||B||)`
- **Range**: 0 a 1 (vetores normalizados)
- **Interpretação**: Mede ângulo entre vetores, independente da magnitude
- **Uso**: Principal métrica para busca por similaridade

### **Distância Euclidiana**
- **Fórmula**: `d(A,B) = √(Σ(ai - bi)²)`
- **Range**: 0 a ∞
- **Interpretação**: Distância "real" no espaço vetorial
- **Uso**: Medida de separação absoluta entre pontos

### **Norma L2**
- **Fórmula**: `||v|| = √(v₁² + v₂² + ... + vₙ²)`
- **Interpretação**: Magnitude/comprimento de um vetor
- **Importância**: Vetores normalizados (norma=1) garantem comparações justas

### **Entropia**
- **Fórmula**: `H = -Σ P(x) × log₂ P(x)`
- **Range**: 0 a log₂(n) bits
- **Interpretação**: Mede diversidade/surpresa da informação
- **Aplicação**: Alta entropia = boa diversidade nos embeddings

---

## 🔍 Métricas de Avaliação

### **Precision (Precisão)**
- **Fórmula**: `Documentos relevantes recuperados / Total de documentos recuperados`
- **Pergunta**: "Dos que recuperei, quantos são realmente relevantes?"
- **Precision@K**: Precisão considerando apenas os K primeiros resultados

### **Recall (Revocação)**
- **Fórmula**: `Documentos relevantes recuperados / Total de documentos relevantes`
- **Pergunta**: "Dos documentos relevantes, quantos consegui recuperar?"
- **Recall@K**: Recall considerando apenas os K primeiros resultados

### **F1-Score**
- **Fórmula**: `2 × (Precision × Recall) / (Precision + Recall)`
- **Interpretação**: Média harmônica entre precisão e recall
- **Uso**: Balanceia ambas as métricas

### **Mean Reciprocal Rank (MRR)**
- **Fórmula**: `(1/n) × Σ(1/rank_i)`
- **Interpretação**: Foca na posição do primeiro resultado relevante
- **Importância**: Crítico para interfaces onde usuário vê poucos resultados

---

## 📈 Conceitos Estatísticos

### **Skewness (Assimetria)**
- **Range**: -∞ a +∞
- **Interpretação**:
  - 0: Distribuição simétrica
  - >0: Cauda à direita (valores altos raros)
  - <0: Cauda à esquerda (valores baixos raros)

### **Kurtosis (Curtose)**
- **Range**: -∞ a +∞
- **Interpretação**:
  - 0: Distribuição normal
  - >0: Caudas pesadas (muitos outliers)
  - <0: Caudas leves (poucos outliers)

### **Z-Score**
- **Fórmula**: `z = (x - μ) / σ`
- **Interpretação**: Quantos desvios padrão um valor está da média
- **Uso**: Detecção de outliers (|z| > 3 é considerado outlier)

### **IQR (Interquartile Range)**
- **Fórmula**: `Q3 - Q1`
- **Uso**: Detecção robusta de outliers
- **Limites**: `Q1 - 1.5×IQR` e `Q3 + 1.5×IQR`

### **Correlação de Pearson**
- **Range**: -1 a +1
- **Interpretação**:
  - |r| > 0.8: Correlação forte
  - |r| 0.5-0.8: Correlação moderada
  - |r| < 0.3: Correlação fraca

---

## 🤖 Algoritmos de Machine Learning

### **K-means Clustering**
- **Objetivo**: Agrupar pontos similares em K clusters
- **Processo**: Minimiza distância intra-cluster, maximiza inter-cluster
- **Parâmetros**: Número de clusters (K)
- **Uso**: Identificar temas/grupos nos documentos

### **PCA (Principal Component Analysis)**
- **Objetivo**: Reduzir dimensionalidade preservando máxima variância
- **Processo**: Encontra direções de maior variação nos dados
- **Uso**: Visualização de embeddings em 2D/3D

### **Isolation Forest**
- **Objetivo**: Detectar outliers através de isolamento
- **Princípio**: Outliers são mais fáceis de isolar (menos partições)
- **Vantagem**: Funciona bem em alta dimensionalidade

---

## 🏗️ Arquitetura RAG

### **Ingestão (Ingestion)**
1. **Loading**: Carregar documentos de fontes
2. **Splitting**: Dividir em chunks
3. **Embedding**: Converter texto em vetores
4. **Indexing**: Armazenar no vector store

### **Recuperação (Retrieval)**
1. **Query Embedding**: Converter pergunta em vetor
2. **Similarity Search**: Buscar chunks similares
3. **Ranking**: Ordenar por relevância
4. **Filtering**: Aplicar filtros adicionais

### **Geração (Generation)**
1. **Context Assembly**: Montar contexto com chunks
2. **Prompt Engineering**: Criar prompt para LLM
3. **Generation**: Gerar resposta
4. **Post-processing**: Refinar saída

---

## 🔧 Parâmetros de Configuração

### **Chunk Size (Tamanho do Chunk)**
- **Pequeno (200-500)**: Alta precisão, pode perder contexto
- **Médio (500-1000)**: Balanceado
- **Grande (1000-2000)**: Mais contexto, pode incluir ruído

### **Chunk Overlap (Sobreposição)**
- **0%**: Sem sobreposição, risco de cortar informação
- **10-20%**: Sobreposição conservadora
- **20-50%**: Sobreposição alta, mais contexto

### **K (Número de Documentos Recuperados)**
- **K=1-3**: Rápido, alta precisão, pode perder contexto
- **K=5-10**: Balanceado para maioria dos casos
- **K=15+**: Muito contexto, possível ruído

### **Similarity Threshold**
- **0.7-0.8**: Threshold conservador
- **0.6-0.7**: Threshold balanceado
- **0.4-0.6**: Threshold liberal

---

## 🎛️ Hiperparâmetros Avançados

### **Top-P (Nucleus Sampling)**
- **Range**: 0.0 a 1.0
- **Interpretação**: Controla diversidade na geração
- **Uso**: Top-P = 0.9 é comum para RAG

### **Temperature**
- **Range**: 0.0 a 2.0+
- **Interpretação**: Controla aleatoriedade
- **Uso**: 0.1-0.3 para RAG (mais determinístico)

### **Max Tokens**
- **Interpretação**: Limite de tokens na resposta
- **Recomendação**: 500-1500 para respostas balanceadas

---

## 📊 Interpretação de Resultados

### **Valores Ideais**

#### **Para Embeddings**:
- Normalização: `is_normalized = true`
- Correlações altas: <10% das dimensões
- Outliers: <5% dos documentos
- Entropia: >3.0 bits

#### **Para Retrieval**:
- Recall@5: >0.8
- Precision@3: >0.7
- MRR: >0.6
- Tempo resposta: <3 segundos

#### **Para Similaridades**:
- Duplicatas: <10% dos pares
- Similaridade média: 0.3-0.6
- Clusters balanceados: Nenhum >60% dos docs

### **Sinais de Problema**

#### **Embeddings Problemáticos**:
- `is_normalized = false`
- Muitas correlações (>20%)
- Muitos outliers (>10%)
- Entropia baixa (<2.0)

#### **Retrieval Ruim**:
- Recall baixo (<0.6)
- Precision baixa (<0.5)
- Tempo alto (>5s)
- MRR baixo (<0.3)

#### **Dataset Problemático**:
- Muitas duplicatas (>20%)
- Um cluster dominante (>80%)
- Similaridade muito alta (>0.8) ou baixa (<0.2)

---

## 🛠️ Ferramentas e Tecnologias

### **LangChain**
Framework Python para desenvolvimento de aplicações com LLMs, incluindo:
- Abstrações para embeddings
- Integrações com vector stores
- Templates para RAG

### **ChromaDB**
- Vector database open-source
- Suporte a múltiplos algoritmos de busca
- Interface Python simples

### **Ollama**
- Platform para rodar LLMs localmente
- Suporte a múltiplos modelos
- API compatível com OpenAI

### **TikToken**
- Tokenizador oficial da OpenAI
- Contagem precisa de tokens
- Suporte a múltiplos modelos

---

## 📚 Modelos de Embedding

### **Nomic Embed Text**
- **Dimensões**: 768
- **Contexto**: 8192 tokens
- **Especialidade**: Textos gerais, boa qualidade

### **BGE (BAAI General Embedding)**
- **Dimensões**: 1024
- **Especialidade**: Multilingual, alta performance

### **E5 (Text Embeddings by Microsoft)**
- **Dimensões**: 1024
- **Especialidade**: Textos técnicos, código

---

## 🔍 Debugging e Troubleshooting

### **Problemas Comuns**

#### **"Embeddings não normalizados"**
- **Causa**: Modelo ou preprocessamento incorreto
- **Solução**: Verificar configuração do modelo, aplicar normalização manual

#### **"Muitas duplicatas encontradas"**
- **Causa**: Dataset mal curado, chunking inadequado
- **Solução**: Limpar dados, ajustar tamanho dos chunks

#### **"Recall muito baixo"**
- **Causa**: Chunks muito específicos, K muito baixo
- **Solução**: Aumentar K, revisar estratégia de chunking

#### **"Tempo de resposta alto"**
- **Causa**: Índice não otimizado, muitos documentos
- **Solução**: Otimizar índice, reduzir K, usar filtering

### **Métricas de Monitoramento**

#### **Em Produção**:
- Tempo de resposta por query
- Taxa de queries sem resultados
- Satisfação do usuário (thumbs up/down)
- Distribuição dos scores de similaridade

#### **Durante Desenvolvimento**:
- Qualidade dos embeddings
- Distribuição das duplicatas
- Performance dos experimentos
- Cobertura temática do dataset

---

Este glossário serve como referência completa para compreender todos os conceitos, métricas e tecnologias envolvidas nos scripts avançados de análise RAG! 📚🚀
