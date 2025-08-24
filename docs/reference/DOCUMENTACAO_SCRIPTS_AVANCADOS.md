# 📚 Documentação Completa dos Scripts Avançados RAG

## 📖 Índice
1. [Visão Geral](#visão-geral)
2. [advanced_metrics.py](#advanced_metricspy)
3. [analyze_similarity.py](#analyze_similaritypy)
4. [evaluate_rag.py](#evaluate_ragpy)
5. [analyze_retrieval.py](#analyze_retrievalpy)
6. [experiment.py](#experimentpy)
7. [Como Usar os Scripts](#como-usar-os-scripts)
8. [Interpretação dos Resultados](#interpretação-dos-resultados)

---

## 🎯 Visão Geral

Os scripts avançados fornecem análises profundas do sistema RAG, permitindo:
- **Avaliação científica** da qualidade dos embeddings
- **Análise matemática** das representações vetoriais
- **Otimização baseada em dados** dos parâmetros do sistema
- **Detecção de problemas** como duplicatas ou outliers
- **Experimentação controlada** com diferentes configurações

### 🔬 Fundamentação Científica

Cada script baseia-se em princípios matemáticos e estatísticos sólidos:
- **Álgebra Linear**: Análise de espaços vetoriais e similaridades
- **Estatística**: Distribuições, outliers e correlações
- **Machine Learning**: Métricas de qualidade e clustering
- **Teoria da Informação**: Entropia e diversidade

---

## 📊 advanced_metrics.py

### 🎯 **Propósito**
Realiza análise matemática profunda dos embeddings para avaliar a qualidade das representações vetoriais.

### 🔧 **Como Funciona**

#### **1. Análise de Qualidade dos Embeddings**
```python
def analyze_embedding_quality(self) -> Dict[str, Any]:
```

**O que faz**: Avalia se os embeddings seguem as melhores práticas matemáticas.

**Métricas Analisadas**:

##### **Normalização**
- **Por que é importante**: Embeddings normalizados garantem que a similaridade cosseno funcione corretamente
- **Como medir**: Calcula a norma L2 de cada vetor (`np.linalg.norm`)
- **Valor ideal**: Norma = 1.0 para todos os vetores
- **Interpretação**:
  - ✅ `is_normalized: true` = Embeddings corretos
  - ❌ `is_normalized: false` = Problema no modelo de embedding

##### **Centralização**
- **Por que é importante**: Embeddings centrados evitam viés direccional
- **Como medir**: Média de todos os valores (`np.mean(embedding_flat)`)
- **Valor ideal**: Próximo de 0
- **Interpretação**:
  - Valor próximo de 0 = Bem balanceado
  - Valor alto = Viés sistemático

##### **Análise de Dimensões**
- **Por que é importante**: Dimensões com baixa variância não contribuem para diferenciação
- **Métricas**:
  - `mean_variance_across_dims`: Variância média entre dimensões
  - `dimensions_with_low_variance`: Dimensões "mortas" (< 0.01)
  - `dimensions_with_high_variance`: Dimensões muito ativas (> 0.1)
- **Interpretação**:
  - Muitas dimensões com baixa variância = Ineficiência
  - Poucas dimensões com alta variância = Concentração de informação

##### **Análise de Correlação**
- **Por que é importante**: Dimensões correlacionadas são redundantes
- **Como medir**: Matriz de correlação entre dimensões
- **Interpretação**:
  - `high_correlation_pairs`: Pares redundantes (correlação > 0.8)
  - Muitas correlações altas = Redundância no espaço vetorial

##### **Distribuição Estatística**
- **Skewness (Assimetria)**: Mede se a distribuição é simétrica
  - Valor = 0: Distribuição simétrica
  - Valor > 0: Cauda à direita
  - Valor < 0: Cauda à esquerda
- **Kurtosis (Curtose)**: Mede "peso" das caudas
  - Valor = 0: Distribuição normal
  - Valor > 0: Caudas pesadas
  - Valor < 0: Caudas leves

#### **2. Análise de Distribuições de Distância**
```python
def analyze_distance_distributions(self) -> Dict[str, Any]:
```

**O que faz**: Analisa como os vetores estão distribuídos no espaço.

**Métricas de Distância**:

##### **Distância Euclidiana**
- **Fórmula**: `√(Σ(ai - bi)²)`
- **Uso**: Distância "real" no espaço vetorial
- **Interpretação**:
  - Média baixa = Documentos similares
  - Desvio padrão alto = Grande diversidade

##### **Distância Cosseno**
- **Fórmula**: `1 - (A·B)/(|A||B|)`
- **Uso**: Ignora magnitude, foca em direção
- **Interpretação**:
  - 0 = Vetores idênticos em direção
  - 1 = Vetores opostos
  - 0.5 = Vetores perpendiculares

##### **Percentis**
- **P25, P75**: Quartis da distribuição
- **P95**: Detecção de outliers
- **Uso**: Compreender a forma da distribuição

#### **3. Análise de Características dos Documentos**
```python
def analyze_document_characteristics(self) -> Dict[str, Any]:
```

**O que faz**: Relaciona propriedades textuais com qualidade dos embeddings.

**Métricas Analisadas**:
- **Tamanho dos documentos**: Correlação entre tamanho e qualidade
- **Contagem de palavras**: Densidade informacional
- **Análise por fonte**: Consistência entre diferentes fontes
- **Similaridade intra-fonte**: Coesão temática

#### **4. Detecção de Outliers**
```python
def detect_outliers(self, method: str = 'isolation_forest') -> Dict[str, Any]:
```

**Métodos Disponíveis**:

##### **Z-Score**
- **Como funciona**: Mede quantos desvios padrão um ponto está da média
- **Threshold**: Normalmente 3.0
- **Uso**: Detecção de pontos anômalos em distribuições normais

##### **IQR (Interquartile Range)**
- **Como funciona**: Usa quartis para definir limites
- **Fórmula**: `Q1 - 1.5×IQR` e `Q3 + 1.5×IQR`
- **Uso**: Mais robusto para distribuições não-normais

**Interpretação dos Outliers**:
- **Poucos outliers**: Sistema saudável
- **Muitos outliers**: Possível problema nos dados ou modelo
- **Outliers específicos**: Documentos únicos ou erros

#### **5. Cálculo de Entropia**
```python
def calculate_embedding_entropy(self) -> Dict[str, float]:
```

**O que é Entropia**: Mede a "surpresa" ou diversidade da informação.

**Fórmula**: `H = -Σ(p(x) × log₂(p(x)))`

**Interpretação**:
- **Alta entropia**: Grande diversidade, boa representação
- **Baixa entropia**: Pouca diversidade, possível problema
- **Entropia por dimensão**: Identifica dimensões informativas

### 📋 **Parâmetros de Uso**

```bash
# Análise completa
python scripts/advanced_metrics.py --all

# Análises específicas
python scripts/advanced_metrics.py --quality      # Qualidade dos embeddings
python scripts/advanced_metrics.py --distances   # Distribuições de distância
python scripts/advanced_metrics.py --documents   # Características dos documentos
python scripts/advanced_metrics.py --outliers z_score  # Detecção de outliers
python scripts/advanced_metrics.py --entropy     # Cálculo de entropia

# Salvar resultados
python scripts/advanced_metrics.py --all --output metrics_report.json
```

### 📈 **Interpretação dos Resultados**

#### **✅ Sistema Saudável**:
- Embeddings normalizados (`is_normalized: true`)
- Baixa correlação entre dimensões
- Poucos outliers
- Entropia balanceada
- Distribuições consistentes

#### **⚠️ Possíveis Problemas**:
- `is_normalized: false` → Problema no modelo
- Muitas `high_correlation_pairs` → Redundância
- Muitos outliers → Dados inconsistentes
- Entropia muito baixa → Pouca diversidade

---

## 📈 analyze_similarity.py

### 🎯 **Propósito**
Analisa similaridades entre chunks para identificar duplicatas, padrões e agrupamentos.

### 🔧 **Como Funciona**

#### **1. Matriz de Similaridade**
```python
def create_similarity_heatmap(embeddings, texts, save_path="similarity_heatmap.png"):
```

**O que faz**: Cria visualização das similaridades entre todos os pares de chunks.

**Como funciona**:
1. **Cálculo**: `cosine_similarity(embeddings)` - matriz NxN
2. **Visualização**: Heatmap com cores representando similaridade
3. **Interpretação**:
   - Verde escuro: Alta similaridade (próximo de 1.0)
   - Verde claro: Similaridade média (0.5-0.8)
   - Amarelo: Baixa similaridade (0.2-0.5)
   - Azul: Muito diferentes (próximo de 0.0)

**Importância**:
- **Identificar duplicatas**: Valores muito altos (>0.9)
- **Encontrar padrões**: Clusters de alta similaridade
- **Avaliar diversidade**: Distribuição das similaridades

#### **2. Detecção de Duplicatas**
```python
def find_duplicates_and_similar(embeddings, texts, threshold=0.9):
```

**O que faz**: Encontra chunks muito similares que podem ser duplicatas.

**Parâmetros**:
- `threshold`: Limite de similaridade (padrão: 0.9)
  - 0.95-1.0: Duplicatas quase exatas
  - 0.9-0.95: Muito similares
  - 0.8-0.9: Similares
  - <0.8: Diferentes

**Interpretação dos Resultados**:
- **Similaridade > 0.95**: Possível duplicata exata
- **Similaridade 0.9-0.95**: Conteúdo muito similar
- **Similaridade 0.8-0.9**: Tema relacionado

**Ações Recomendadas**:
- Duplicatas exatas: Remover para eficiência
- Muito similares: Avaliar necessidade
- Similares: Podem ser úteis para contexto

#### **3. Clustering de Documentos**
```python
def cluster_documents(embeddings, texts, n_clusters=3, save_path="clusters_plot.png"):
```

**O que faz**: Agrupa documentos similares usando K-means clustering.

**Como funciona**:
1. **K-means**: Algoritmo de clustering não-supervisionado
2. **PCA**: Redução para 2D para visualização
3. **Visualização**: Scatter plot com cores por cluster

**Interpretação**:
- **Clusters bem separados**: Temas distintos
- **Clusters sobrepostos**: Temas relacionados
- **Pontos isolados**: Conteúdo único

**Análise de Clusters**:
- **Tamanho dos clusters**: Distribuição dos temas
- **Coesão interna**: Qualidade do agrupamento
- **Separação entre clusters**: Diversidade temática

#### **4. Estatísticas dos Embeddings**
```python
def analyze_embedding_statistics(embeddings):
```

**Métricas Calculadas**:

##### **Estatísticas Básicas**:
- **Forma**: Dimensões da matriz (n_docs × n_dimensions)
- **Média geral**: Valor médio de todos os elementos
- **Desvio padrão**: Variabilidade dos valores
- **Min/Max**: Amplitude dos valores

##### **Normas L2**:
- **O que é**: Magnitude de cada vetor embedding
- **Fórmula**: `√(x₁² + x₂² + ... + xₙ²)`
- **Interpretação**:
  - Norma ≈ 1.0: Vetor normalizado ✅
  - Variação alta: Possível problema

##### **Similaridade Média**:
- **Cálculo**: Média da matriz de similaridade (excluindo diagonal)
- **Interpretação**:
  - Baixa (0.0-0.3): Documentos muito diversos
  - Média (0.3-0.7): Diversidade balanceada
  - Alta (0.7-1.0): Documentos muito similares

#### **5. Visualização de Dimensões**
```python
def visualize_embedding_dimensions(embeddings, save_path="embedding_dimensions.png"):
```

**O que faz**: Analisa como cada dimensão do embedding se comporta.

**Gráficos Gerados**:
1. **Média por dimensão**: Mostra viés de cada dimensão
2. **Desvio padrão por dimensão**: Mostra variabilidade

**Interpretação**:
- **Dimensões com baixo desvio**: Pouco informativas
- **Dimensões com alto desvio**: Muito informativas
- **Padrões**: Podem indicar estrutura no modelo

### 📋 **Parâmetros de Uso**

```bash
# Análise completa
python scripts/analyze_similarity.py --all

# Análises específicas
python scripts/analyze_similarity.py --stats                    # Estatísticas básicas
python scripts/analyze_similarity.py --heatmap                 # Criar heatmap
python scripts/analyze_similarity.py --duplicates 0.9          # Encontrar duplicatas
python scripts/analyze_similarity.py --clusters 5              # Clustering (5 grupos)
python scripts/analyze_similarity.py --dimensions              # Análise de dimensões

# Combinações
python scripts/analyze_similarity.py --duplicates 0.8 --clusters 3
```

### 📈 **Interpretação dos Resultados**

#### **🔍 Análise de Duplicatas**:
- **0 duplicatas**: ✅ Boa diversidade
- **Poucas duplicatas (1-2)**: ⚠️ Verificar se são relevantes
- **Muitas duplicatas (>20%)**: ❌ Problema nos dados

#### **🎭 Análise de Clusters**:
- **Clusters balanceados**: ✅ Boa distribuição temática
- **Um cluster dominante**: ⚠️ Falta diversidade
- **Muitos clusters pequenos**: ⚠️ Fragmentação excessiva

#### **📊 Estatísticas**:
- **Similaridade média 0.3-0.6**: ✅ Balanceado
- **Similaridade média >0.8**: ❌ Muito homogêneo
- **Similaridade média <0.2**: ❌ Muito fragmentado

---

## ⚖️ evaluate_rag.py

### 🎯 **Propósito**
Avalia a qualidade end-to-end do sistema RAG com métricas científicas.

### 🔧 **Como Funciona**

#### **1. Métricas de Avaliação**

##### **Similaridade de Recuperação**
```python
similarity_score = cosine_similarity([query_embedding], retrieved_embeddings)[0]
```
- **O que mede**: Quão bem os chunks recuperados relacionam-se com a query
- **Interpretação**:
  - >0.8: Excelente recuperação
  - 0.6-0.8: Boa recuperação
  - 0.4-0.6: Recuperação regular
  - <0.4: Recuperação ruim

##### **Relevância Semântica**
- **Como medir**: Análise da sobreposição de temas/conceitos
- **Métricas**: Baseada em keywords e entidades compartilhadas
- **Importância**: Mede se o conteúdo é realmente útil

##### **Faithfulness (Fidelidade)**
- **O que é**: Se a resposta é consistente com o contexto recuperado
- **Como medir**: Análise de contradições e informações não suportadas
- **Importância**: Evita "alucinações" do modelo

##### **Tempo de Resposta**
- **Componentes**:
  - Tempo de recuperação (busca vetorial)
  - Tempo de geração (LLM)
  - Tempo total
- **Benchmarks**:
  - <1s: Excelente
  - 1-3s: Bom
  - 3-5s: Aceitável
  - >5s: Lento

#### **2. Processo de Avaliação**

```python
def evaluate_query(self, query: str, expected_answer: str = None) -> Dict[str, Any]:
```

**Fluxo**:
1. **Embedding da query**: Converte pergunta em vetor
2. **Recuperação**: Busca chunks similares
3. **Geração**: Cria resposta com LLM
4. **Análise**: Calcula métricas de qualidade

**Métricas Retornadas**:
- `similarity_scores`: Similaridades com chunks recuperados
- `relevance_score`: Relevância semântica
- `response_time`: Tempo total de processamento
- `retrieval_time`: Tempo apenas da busca
- `generation_time`: Tempo apenas da geração

### 📋 **Parâmetros de Uso**

```bash
# Avaliação com query específica
python scripts/evaluate_rag.py --query "Como funciona o sistema de login?"

# Avaliação em lote
python scripts/evaluate_rag.py --batch

# Salvar resultados
python scripts/evaluate_rag.py --query "..." --output evaluation.json

# Comparar configurações
python scripts/evaluate_rag.py --compare-k 3 5 10
```

### 📈 **Interpretação dos Resultados**

#### **✅ Sistema Funcionando Bem**:
- Similaridade média >0.6
- Tempo de resposta <3s
- Respostas consistentes com contexto
- Alta diversidade nos chunks recuperados

#### **⚠️ Possíveis Melhorias**:
- Similaridade 0.4-0.6: Melhorar embeddings ou chunks
- Tempo >5s: Otimizar indexação ou modelo
- Baixa relevância: Revisar estratégia de chunking

---

## 🔍 analyze_retrieval.py

### 🎯 **Propósito**
Analisa especificamente a qualidade do sistema de recuperação (busca vetorial).

### 🔧 **Como Funciona**

#### **1. Métricas de Recuperação**

##### **Recall@K**
- **Fórmula**: `Documentos relevantes recuperados / Total de documentos relevantes`
- **Interpretação**:
  - Recall@3 = 0.8: 80% dos documentos relevantes estão nos top-3
  - Recall@10 = 1.0: Todos os documentos relevantes estão nos top-10

##### **Precision@K**
- **Fórmula**: `Documentos relevantes recuperados / K documentos recuperados`
- **Interpretação**:
  - Precision@5 = 0.6: 60% dos 5 documentos retornados são relevantes
  - Precision@1 = 1.0: O primeiro resultado é sempre relevante

##### **Mean Reciprocal Rank (MRR)**
- **Fórmula**: `1 / posição do primeiro resultado relevante`
- **Interpretação**:
  - MRR = 1.0: Primeiro resultado sempre relevante
  - MRR = 0.5: Primeiro resultado relevante na posição 2 em média

#### **2. Análise de Distribuição de Scores**

```python
def analyze_score_distribution(self, queries: List[str]) -> Dict[str, Any]:
```

**O que analisa**:
- **Distribuição das similaridades**: Como os scores se distribuem
- **Gap entre top results**: Diferença entre melhor e segundo melhor
- **Threshold analysis**: Qual score mínimo garantir qualidade

**Interpretação**:
- **Alto gap**: Boa discriminação entre relevante/irrelevante
- **Baixo gap**: Dificuldade em distinguir relevância
- **Distribuição uniforme**: Possível problema no modelo

#### **3. Detecção de Queries Problemáticas**

```python
def identify_problematic_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
```

**Identifica**:
- **Queries com baixo score máximo**: Nenhum documento muito relevante
- **Queries com alta variância**: Resultados inconsistentes
- **Queries sem resultados**: Falhas na recuperação

### 📋 **Parâmetros de Uso**

```bash
# Análise completa do sistema de recuperação
python scripts/analyze_retrieval.py --full-analysis

# Análise de queries específicas
python scripts/analyze_retrieval.py --queries "login" "sistema" "otimização"

# Análise de recall/precision
python scripts/analyze_retrieval.py --metrics recall precision mrr

# Comparar diferentes valores de K
python scripts/analyze_retrieval.py --compare-k 1 3 5 10
```

---

## 🧪 experiment.py

### 🎯 **Propósito**
Framework para experimentação controlada com diferentes configurações do RAG.

### 🔧 **Como Funciona**

#### **1. Experimentação de Chunk Size**

```python
def experiment_chunk_sizes(self, sizes: List[int], queries: List[str]) -> Dict[str, Any]:
```

**O que testa**: Impacto do tamanho dos chunks na qualidade da recuperação.

**Tamanhos típicos**:
- **200-500**: Chunks pequenos, alta precisão
- **500-1000**: Balanceado
- **1000-2000**: Chunks grandes, mais contexto

**Métricas avaliadas**:
- Qualidade da recuperação
- Tempo de processamento
- Cobertura de informação

#### **2. Experimentação de K (número de documentos)**

```python
def experiment_retrieval_k(self, k_values: List[int], queries: List[str]) -> Dict[str, Any]:
```

**O que testa**: Quantos documentos recuperar para otimizar qualidade vs. velocidade.

**Trade-offs**:
- **K baixo (1-3)**: Rápido, mas pode perder contexto
- **K médio (5-10)**: Balanceado
- **K alto (15+)**: Mais contexto, mas mais ruído

#### **3. Comparação A/B**

```python
def run_ab_test(self, config_a: Dict, config_b: Dict, queries: List[str]) -> Dict[str, Any]:
```

**Permite comparar**:
- Diferentes modelos de embedding
- Diferentes estratégias de chunking
- Diferentes parâmetros de busca

### 📋 **Parâmetros de Uso**

```bash
# Experimentar tamanhos de chunk
python scripts/experiment.py --chunk-sizes 200 500 1000

# Experimentar valores de K
python scripts/experiment.py --k-values 1 3 5 10

# Teste A/B completo
python scripts/experiment.py --ab-test --config-a config_a.json --config-b config_b.json

# Experimento completo
python scripts/experiment.py --full-experiment
```

---

## 🚀 Como Usar os Scripts

### 📋 **Fluxo Recomendado de Análise**

#### **1. Análise Inicial (Saúde do Sistema)**
```bash
# Verificar qualidade básica dos embeddings
python scripts/advanced_metrics.py --quality --documents

# Verificar duplicatas e similaridades
python scripts/analyze_similarity.py --stats --duplicates
```

#### **2. Análise Profunda**
```bash
# Análise matemática completa
python scripts/advanced_metrics.py --all --output metrics_full.json

# Análise visual de similaridades
python scripts/analyze_similarity.py --all
```

#### **3. Avaliação de Performance**
```bash
# Testar qualidade end-to-end
python scripts/evaluate_rag.py --query "Como funciona o login?"

# Analisar sistema de recuperação
python scripts/analyze_retrieval.py --full-analysis
```

#### **4. Experimentação e Otimização**
```bash
# Experimentar diferentes configurações
python scripts/experiment.py --chunk-sizes 200 500 1000
python scripts/experiment.py --k-values 3 5 10
```

### 🔧 **Instalação de Dependências**

```bash
# Dependências para análise matemática
pip install scipy numpy

# Dependências para visualização
pip install matplotlib seaborn

# Dependências para machine learning
pip install scikit-learn

# Ou instalar tudo de uma vez
pip install scipy numpy matplotlib seaborn scikit-learn
```

---

## 📈 Interpretação dos Resultados

### 🎯 **Indicadores de Sistema Saudável**

#### **Embeddings de Qualidade**:
- ✅ `is_normalized: true`
- ✅ Similaridade média entre 0.3-0.6
- ✅ Poucos outliers (<5%)
- ✅ Entropia balanceada
- ✅ Baixa correlação entre dimensões

#### **Recuperação Eficiente**:
- ✅ Recall@5 > 0.8
- ✅ Precision@3 > 0.7
- ✅ Tempo de resposta < 3s
- ✅ Poucos gaps grandes entre scores

#### **Diversidade Adequada**:
- ✅ Clusters bem balanceados
- ✅ Poucas duplicatas (<10%)
- ✅ Boa distribuição temática

### ⚠️ **Sinais de Problemas**

#### **Problemas nos Embeddings**:
- ❌ `is_normalized: false` → Corrigir modelo
- ❌ Muitas correlações altas → Redundância
- ❌ Muitos outliers → Dados inconsistentes

#### **Problemas na Recuperação**:
- ❌ Recall baixo → Melhorar indexação
- ❌ Precision baixa → Ajustar chunking
- ❌ Tempo alto → Otimizar busca

#### **Problemas nos Dados**:
- ❌ Muitas duplicatas → Limpar dataset
- ❌ Clusters desequilibrados → Balancear conteúdo
- ❌ Baixa diversidade → Expandir fontes

### 🛠️ **Ações Corretivas**

#### **Para Melhorar Embeddings**:
1. Verificar normalização do modelo
2. Aumentar dimensionalidade se necessário
3. Treinar modelo específico para domínio

#### **Para Melhorar Recuperação**:
1. Ajustar tamanho dos chunks
2. Otimizar número de documentos recuperados (K)
3. Implementar re-ranking
4. Usar híbrido (busca vetorial + texto)

#### **Para Melhorar Dados**:
1. Remover duplicatas
2. Balancear distribuição temática
3. Adicionar mais fontes diversas
4. Melhorar qualidade do chunking

---

## 🎓 **Conceitos Educacionais**

### 📐 **Matemática por trás dos Embeddings**

#### **Similaridade Cosseno**:
```
cos(θ) = (A · B) / (|A| × |B|)
```
- Mede ângulo entre vetores
- Independe da magnitude
- Valor entre -1 e 1 (normalizado: 0 a 1)

#### **Distância Euclidiana**:
```
d(A,B) = √(Σ(ai - bi)²)
```
- Distância "real" no espaço
- Sensível à magnitude
- Sempre positiva

#### **Entropia**:
```
H(X) = -Σ P(x) × log₂ P(x)
```
- Mede "surpresa" da informação
- Alta entropia = mais diversidade
- Baixa entropia = mais previsibilidade

### 🧠 **Conceitos de Machine Learning**

#### **K-means Clustering**:
- Agrupa pontos similares
- Minimiza distância intra-cluster
- Maximiza distância inter-cluster

#### **PCA (Principal Component Analysis)**:
- Reduz dimensionalidade
- Preserva máxima variância
- Útil para visualização

#### **Outlier Detection**:
- **Z-score**: Baseado em desvios padrão
- **IQR**: Baseado em quartis
- **Isolation Forest**: Baseado em isolamento

### 📊 **Métricas de Avaliação**

#### **Precision vs Recall**:
- **Precision**: "Dos que recuperei, quantos são relevantes?"
- **Recall**: "Dos relevantes, quantos recuperei?"
- **F1-Score**: Média harmônica entre precision e recall

#### **Mean Reciprocal Rank (MRR)**:
- Foca na posição do primeiro resultado relevante
- Importante para sistemas onde usuário vê poucos resultados

---

Esta documentação fornece uma base sólida para compreender, usar e interpretar todos os scripts avançados do sistema RAG, combinando teoria, prática e interpretação de resultados de forma educacional e aplicável.
