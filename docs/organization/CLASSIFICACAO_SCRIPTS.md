# 📊 Classificação de Scripts por Nível de Complexidade

> **Sistema de Classificação Educacional para Scripts RAG**
> 
> Esta classificação organiza todos os scripts em três níveis de complexidade, facilitando uma progressão estruturada de aprendizagem e definindo pré-requisitos claros.

# 📊 Classificação de Scripts por Nível de Complexidade

> **Sistema de Classificação Educacional para Scripts RAG**
> 
> Esta classificação organiza todos os scripts em três níveis de complexidade, facilitando uma progressão estruturada de aprendizagem e definindo pré-requisitos claros.

---

## 🏗️ **ARQUITETURA COMPLETA DO SISTEMA**

### **📦 Core Package (rag_demo/)**
> **Módulos fundamentais com documentação educacional avançada**

**🔧 Componentes Principais:**
- **`__init__.py`** - Arquitetura de pacote e design patterns
- **`config.py`** - Configuração baseada em ambiente
- **`ingest.py`** - Pipeline de ingestão com fundamentos matemáticos  
- **`rag.py`** - Arquitetura RAG completa (retrieval + generation)
- **`utils.py`** - Funções utilitárias e manipulação de documentos

**📚 Características:**
- Documentação educacional completa
- Explicações de fundamentação técnica
- Padrões de arquitetura aplicados
- Conceitos matemáticos detalhados

---

## 🎯 **CRITÉRIOS DE CLASSIFICAÇÃO**

### **Dimensões Avaliadas**
- **🧠 Complexidade Conceitual:** Conceitos teóricos necessários
- **⚙️ Complexidade Técnica:** Dificuldade de implementação
- **📊 Complexidade Analítica:** Sofisticação das análises
- **🎓 Pré-requisitos:** Conhecimentos necessários
- **⏱️ Tempo de Aprendizagem:** Estimativa de estudo

---

## 🟢 **NÍVEL BÁSICO** - Fundamentos Essenciais

> **Objetivo:** Compreender e operar funcionalidades básicas do sistema RAG
> **Tempo Total:** 2-4 horas
> **Pré-requisito:** Conhecimento básico de Python

### **Scripts Básicos**

#### **🥇 [run_ingest.py](scripts/run_ingest.py)** - *Pipeline de Ingestão*
- **Complexidade:** ⭐⭐☆☆☆
- **Conceitos:** Chunking, Embeddings, Persistência
- **Pré-requisitos:** Conceitos básicos de NLP
- **Tempo:** 30-45 min
- **Por que Básico:** Operação fundamental, interface simples, conceitos diretos
- **Saídas:** Banco vetorial populado, métricas básicas de ingestão

#### **🥈 [run_query.py](scripts/run_query.py)** - *Interface de Consulta*
- **Complexidade:** ⭐⭐☆☆☆
- **Conceitos:** Query Processing, Retrieval, Generation
- **Pré-requisitos:** run_ingest.py executado
- **Tempo:** 30-45 min
- **Por que Básico:** Interface direta, fluxo linear, feedback imediato
- **Saídas:** Respostas contextualizadas, tempo de resposta

#### **🥉 [list_docs.py](scripts/list_docs.py)** - *Exploração de Documentos*
- **Complexidade:** ⭐☆☆☆☆
- **Conceitos:** Metadados, Indexação, Organização
- **Pré-requisitos:** run_ingest.py executado
- **Tempo:** 15-30 min
- **Por que Básico:** Operação de listagem simples, sem processamento complexo
- **Saídas:** Lista de documentos, metadados, estatísticas básicas

#### **4️⃣ [search_docs.py](scripts/search_docs.py)** - *Busca Semântica*
- **Complexidade:** ⭐⭐☆☆☆
- **Conceitos:** Similaridade Semântica, Ranking, Filtragem
- **Pré-requisitos:** run_ingest.py executado, conceitos de similaridade
- **Tempo:** 45-60 min
- **Por que Básico:** Interface intuitiva, conceitos fundamentais de busca
- **Saídas:** Documentos relevantes, scores de similaridade

### **Competências Desenvolvidas - Nível Básico**
- ✅ Compreensão do pipeline RAG completo
- ✅ Operação de sistemas de embeddings
- ✅ Interpretação de métricas básicas
- ✅ Uso de interfaces de consulta
- ✅ Navegação em bases de dados vetoriais

---

## 🟡 **NÍVEL INTERMEDIÁRIO** - Análise e Compreensão

> **Objetivo:** Analisar comportamento do sistema e interpretar métricas de qualidade
> **Tempo Total:** 3-5 horas
> **Pré-requisito:** Todos os scripts básicos executados

### **Scripts Intermediários**

#### **🎯 [analyze_chunks.py](scripts/analyze_chunks.py)** - *Análise de Fragmentação*
- **Complexidade:** ⭐⭐⭐☆☆
- **Conceitos:** Distribuição Estatística, Sobreposição, Otimização
- **Pré-requisitos:** Estatística básica, conceitos de chunking
- **Tempo:** 60-90 min
- **Por que Intermediário:** Análise estatística, interpretação de distribuições
- **Saídas:** Histogramas, métricas de distribuição, recomendações

#### **🎨 [show_vectors.py](scripts/show_vectors.py)** - *Visualização de Embeddings*
- **Complexidade:** ⭐⭐⭐☆☆
- **Conceitos:** Redução de Dimensionalidade, PCA, Clustering
- **Pré-requisitos:** Álgebra linear básica, conceitos de machine learning
- **Tempo:** 60-90 min
- **Por que Intermediário:** Matemática de dimensionalidade, interpretação visual
- **Saídas:** Plots 2D/3D, clusters identificados, métricas de qualidade

#### **📄 [list_raw.py](scripts/list_raw.py)** - *Análise de Dados Brutos*
- **Complexidade:** ⭐⭐☆☆☆
- **Conceitos:** Preprocessamento, Qualidade de Dados, Estrutura
- **Pré-requisitos:** Conceitos de qualidade de dados
- **Tempo:** 45-60 min
- **Por que Intermediário:** Análise de qualidade, diagnóstico de problemas
- **Saídas:** Relatórios de qualidade, estatísticas de conteúdo

### **Competências Desenvolvidas - Nível Intermediário**
- ✅ Análise estatística de dados textuais
- ✅ Interpretação de visualizações de alta dimensionalidade
- ✅ Diagnóstico de problemas de qualidade
- ✅ Otimização de parâmetros de chunking
- ✅ Compreensão de métricas de clustering

---

## 🔴 **NÍVEL AVANÇADO** - Pesquisa e Experimentação

> **Objetivo:** Conduzir pesquisa científica rigorosa e experimentação controlada
> **Tempo Total:** 8-15 horas
> **Pré-requisito:** Domínio dos níveis anteriores + conhecimento em estatística/ML

### **Scripts Avançados**

#### **📊 [advanced_metrics.py](scripts/advanced_metrics.py)** - *Métricas Científicas*
- **Complexidade:** ⭐⭐⭐⭐☆
- **Conceitos:** Estatística Avançada, Normalização, Entropia, Correlação
- **Pré-requisitos:** Estatística inferencial, teoria da informação
- **Tempo:** 2-3 horas
- **Por que Avançado:** Matemática sofisticada, interpretação científica
- **Saídas:** Métricas de qualidade científica, análises estatísticas rigorosas

#### **🔍 [analyze_similarity.py](scripts/analyze_similarity.py)** - *Análise de Similaridade*
- **Complexidade:** ⭐⭐⭐⭐☆
- **Conceitos:** Clustering Avançado, Detecção de Padrões, Heatmaps
- **Pré-requisitos:** Machine learning, análise multivariada
- **Tempo:** 2-3 horas
- **Por que Avançado:** Algoritmos ML complexos, interpretação multidimensional
- **Saídas:** Heatmaps de similaridade, clusters hierárquicos, métricas de separação

#### **🎯 [analyze_retrieval.py](scripts/analyze_retrieval.py)** - *Análise de Recuperação*
- **Complexidade:** ⭐⭐⭐⭐☆
- **Conceitos:** Information Retrieval, Precision/Recall, NDCG, MAP
- **Pré-requisitos:** Information Retrieval theory, métricas de ranking
- **Tempo:** 2-3 horas
- **Por que Avançado:** Teoria de IR, métricas sofisticadas de ranking
- **Saídas:** Curvas precision-recall, métricas de ranking, análise de eficácia

#### **📈 [evaluate_rag.py](scripts/evaluate_rag.py)** - *Avaliação Científica Completa*
- **Complexidade:** ⭐⭐⭐⭐⭐
- **Conceitos:** Benchmarking, Validação Cruzada, Significância Estatística
- **Pré-requisitos:** Metodologia científica, design experimental
- **Tempo:** 3-4 horas
- **Por que Avançado:** Metodologia científica rigorosa, validação estatística
- **Saídas:** Relatórios científicos completos, benchmarks, validação estatística

#### **🧪 [experiment.py](scripts/experiment.py)** - *Framework Experimental*
- **Complexidade:** ⭐⭐⭐⭐⭐
- **Conceitos:** Design Experimental, A/B Testing, Otimização de Hiperparâmetros
- **Pré-requisitos:** Design experimental, otimização, análise multivariada
- **Tempo:** 4-6 horas
- **Por que Avançado:** Framework completo de experimentação científica
- **Saídas:** Experimentos controlados, otimização automática, relatórios científicos

### **Competências Desenvolvidas - Nível Avançado**
- ✅ Condução de pesquisa científica rigorosa
- ✅ Design e execução de experimentos controlados
- ✅ Análise estatística avançada e interpretação
- ✅ Otimização sistemática de hiperparâmetros
- ✅ Validação estatística de hipóteses
- ✅ Publicação de resultados científicos

---

## 📈 **PROGRESSÃO DE APRENDIZAGEM**

### **Curva de Complexidade**
```
Básico (🟢)     Intermediário (🟡)     Avançado (🔴)
     │                    │                    │
     ▼                    ▼                    ▼
Conceitos ────────► Análise ────────► Pesquisa
Simples           Estatística         Científica
     │                    │                    │
2-4 horas          3-5 horas          8-15 horas
```

### **Dependências Entre Níveis**
- **🟢 → 🟡:** Compreensão sólida de operações básicas
- **🟡 → 🔴:** Capacidade de análise estatística e interpretação
- **🔴:** Autonomia para pesquisa científica independente

### **Marcos de Progressão**
1. **🎓 Básico Completo:** Capaz de operar sistema RAG independentemente
2. **📊 Intermediário Completo:** Capaz de analisar e otimizar performance
3. **🔬 Avançado Completo:** Capaz de conduzir pesquisa científica original

---

## 🎯 **RECOMENDAÇÕES POR PERFIL**

### **👨‍🎓 Estudante/Iniciante**
- **Foco:** Nível Básico completo
- **Tempo:** 1-2 semanas (2h/dia)
- **Objetivo:** Compreensão operacional

### **👨‍💻 Desenvolvedor/Engenheiro**
- **Foco:** Básico + Intermediário
- **Tempo:** 2-3 semanas (2-3h/dia)
- **Objetivo:** Implementação e otimização

### **👨‍🔬 Pesquisador/Cientista**
- **Foco:** Todos os níveis
- **Tempo:** 4-6 semanas (3-4h/dia)
- **Objetivo:** Pesquisa e inovação

### **👨‍💼 Gestor/Líder Técnico**
- **Foco:** Básico + overview Avançado
- **Tempo:** 1-2 semanas (1-2h/dia)
- **Objetivo:** Compreensão estratégica

---

## 📋 **MATRIZ DE PRÉ-REQUISITOS DETALHADA**

| Script | Python | Estatística | ML/AI | Math | IR Theory | Exp Design |
|--------|---------|-------------|--------|------|-----------|------------|
| run_ingest.py | ⭐⭐ | - | ⭐ | - | - | - |
| run_query.py | ⭐⭐ | - | ⭐ | - | ⭐ | - |
| list_docs.py | ⭐ | - | - | - | - | - |
| search_docs.py | ⭐⭐ | ⭐ | ⭐ | ⭐ | ⭐ | - |
| analyze_chunks.py | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐ | - | - |
| show_vectors.py | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐ | - | - |
| list_raw.py | ⭐⭐ | ⭐ | - | - | - | - |
| advanced_metrics.py | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐ |
| analyze_similarity.py | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐ |
| analyze_retrieval.py | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| evaluate_rag.py | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| experiment.py | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |

**Legenda:** ⭐ = Básico, ⭐⭐ = Intermediário, ⭐⭐⭐ = Avançado, ⭐⭐⭐⭐ = Expert

---

## 🚀 **PRÓXIMOS PASSOS POR NÍVEL**

### **Após Básico 🟢**
1. **Autoavaliação:** Consegue executar todos os scripts básicos independentemente?
2. **Teste:** Implemente uma variação simples de um script básico
3. **Progressão:** Inicie nível intermediário com analyze_chunks.py

### **Após Intermediário 🟡**
1. **Autoavaliação:** Consegue interpretar e otimizar métricas de qualidade?
2. **Teste:** Conduza uma análise comparativa entre diferentes configurações
3. **Progressão:** Inicie pesquisa avançada com advanced_metrics.py

### **Após Avançado 🔴**
1. **Autoavaliação:** Consegue conduzir pesquisa científica independente?
2. **Teste:** Publique resultados de um experimento original
3. **Progressão:** Contribua com novos scripts ou melhorias ao framework

---

*📊 **Nota Metodológica:** Esta classificação é baseada em análise da complexidade conceitual, técnica e analítica de cada script, bem como feedback de usuários em diferentes estágios de aprendizagem.*
