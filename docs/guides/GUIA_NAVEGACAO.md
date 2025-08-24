# 🧭 Guia de Navegação - Sistema Educacional RAG

> **Mapa completo de navegação entre todos os documentos e scripts**
> 
> Este guia facilita a movimentação entre diferentes partes do sistema educacional, mostrando conexões lógicas e fluxos de aprendizagem.

---

## 🎯 **PONTOS DE ENTRADA PRINCIPAIS**

### 🏗️ **Para Arquitetos de Software**
```
📦 Core Package (rag_demo/) → 🔧 Padrões de Design → 📚 Documentação Técnica
```

### 👤 **Para Novos Usuários**
```
📚 README.md → 🎓 TUTORIAL_RAG.md → 🟢 Scripts Básicos
```

### 🔬 **Para Pesquisadores**
```
📊 DOCUMENTACAO_SCRIPTS_AVANCADOS.md → 🔴 Scripts Avançados → 🧪 experiment.py
```

### 🛠️ **Para Desenvolvedores**
```
📋 DOCUMENTACAO_SCRIPTS_AVANCADOS.md → ⚙️ Código fonte → 🔧 Scripts de apoio
```

---

## 🔄 **FLUXOS DE NAVEGAÇÃO POR OBJETIVO**

### 🏗️ **COMPREENSÃO ARQUITETURAL**
> **Objetivo:** Entender a arquitetura interna do sistema

**🔄 Fluxo Core Package:**
```
1️⃣ rag_demo/config.py ➜ Configuração e ambiente
2️⃣ rag_demo/utils.py ➜ Manipulação de documentos
3️⃣ rag_demo/ingest.py ➜ Pipeline de processamento  
4️⃣ rag_demo/rag.py ➜ Arquitetura RAG completa
5️⃣ rag_demo/__init__.py ➜ Design patterns
```

**📚 Competências Desenvolvidas:**
- Padrões de arquitetura de software
- Configuração baseada em ambiente (twelve-factor app)
- Pipeline de processamento de dados
- Fundamentos matemáticos de embeddings
- Design patterns (Strategy, Factory, Observer)

### 📚 **APRENDIZAGEM PROGRESSIVA**

#### **Trilha Básica (2-4 horas)**
1. **[📖 README.md](../../README.md)** - Visão geral e setup
   - ➡️ **[🎓 TUTORIAL_RAG.md](TUTORIAL_RAG.md)** - Fundamentos teóricos
   - ➡️ **[📝 GLOSSARIO_CONCEITOS.md](../reference/GLOSSARIO_CONCEITOS.md)** - Vocabulário técnico

2. **[🟢 run_ingest.py](scripts/run_ingest.py)** - Primeira ingestão
   - ➡️ **[🟢 run_query.py](scripts/run_query.py)** - Primeiras consultas
   - ➡️ **[🟢 list_docs.py](scripts/list_docs.py)** - Exploração do índice

3. **[📋 EXEMPLOS_USO_SCRIPTS.md](EXEMPLOS_USO_SCRIPTS.md)** - Casos práticos

#### **Trilha Intermediária (4-6 horas)**
1. **[🟡 analyze_chunks.py](scripts/analyze_chunks.py)** - Análise de fragmentação
   - ➡️ **[🟡 show_vectors.py](scripts/show_vectors.py)** - Visualização de embeddings
   - ➡️ **[🟡 list_raw.py](scripts/list_raw.py)** - Dados brutos

2. **[🟢 search_docs.py](scripts/search_docs.py)** - Busca semântica avançada

#### **Trilha Avançada (8-12 horas)**
1. **[📊 DOCUMENTACAO_SCRIPTS_AVANCADOS.md](../reference/DOCUMENTACAO_SCRIPTS_AVANCADOS.md)** - Preparação teórica
2. **[🔴 advanced_metrics.py](scripts/advanced_metrics.py)** - Métricas científicas
3. **[🔴 analyze_similarity.py](scripts/analyze_similarity.py)** - Análise de similaridade
4. **[🔴 analyze_retrieval.py](scripts/analyze_retrieval.py)** - Avaliação de recuperação
5. **[🔴 evaluate_rag.py](scripts/evaluate_rag.py)** - Avaliação completa
6. **[🧪 experiment.py](scripts/experiment.py)** - Experimentação científica

#### **🛡️ Trilha Guardrails e Produção (3-5 horas)**
1. **[🛡️ USO_GUARDRAILS.md](USO_GUARDRAILS.md)** - Guia prático de uso
2. **[� THRESHOLD_OPTIMIZER_GUIDE.md](../reference/THRESHOLD_OPTIMIZER_GUIDE.md)** - Guia completo do optimizer
3. **[�🛡️ rag_with_guardrails.py](scripts/rag_with_guardrails.py)** - RAG seguro
4. **[📊 threshold_optimizer.py](scripts/threshold_optimizer.py)** - Otimização automática
5. **[📚 GUARDRAILS_GUIDE.md](../reference/GUARDRAILS_GUIDE.md)** - Documentação técnica completa

### 🎯 **NAVEGAÇÃO POR PROBLEMA ESPECÍFICO**

#### **Problema: "Resultados de busca irrelevantes"**
```
🔍 Diagnóstico:
search_docs.py → analyze_retrieval.py → evaluate_rag.py

📊 Análise:
advanced_metrics.py → analyze_similarity.py

🛠️ Otimização:
experiment.py → analyze_chunks.py
```

#### **Problema: "Chunks muito pequenos/grandes"**
```
📊 Análise:
analyze_chunks.py → list_raw.py

🔧 Experimentação:
experiment.py (chunk size experiments)

✅ Validação:
advanced_metrics.py → evaluate_rag.py
```

#### **Problema: "Embeddings de baixa qualidade"**
```
🎨 Visualização:
show_vectors.py → analyze_similarity.py

📊 Métricas:
advanced_metrics.py

🧪 Comparação:
experiment.py (model comparison)
```

### 🔬 **NAVEGAÇÃO POR PESQUISA CIENTÍFICA**

#### **Pesquisa: Otimização de Hiperparâmetros**
```
📋 Planejamento:
DOCUMENTACAO_SCRIPTS_AVANCADOS.md

🧪 Experimentação:
experiment.py → advanced_metrics.py

📊 Avaliação:
evaluate_rag.py → analyze_retrieval.py

📝 Documentação:
GLOSSARIO_CONCEITOS.md (novos termos)
```

#### **Pesquisa: Comparação de Modelos**
```
🎯 Setup:
experiment.py (model comparison)

📊 Métricas:
advanced_metrics.py → analyze_similarity.py

🔍 Análise:
analyze_retrieval.py → evaluate_rag.py

📈 Visualização:
show_vectors.py
```

---

## 🔗 **MATRIZ DE INTERCONEXÕES**

### **Documentos de Entrada**
| Documento | Leva Para | Propósito |
|-----------|-----------|-----------|
| **README.md** | TUTORIAL_RAG.md, Scripts Básicos | Orientação inicial |
| **INDICE_DOCUMENTACAO.md** | Todos os documentos | Navegação estruturada |
| **TUTORIAL_RAG.md** | Scripts práticos | Fundamentação teórica |

### **Documentos de Apoio**
| Documento | Usado Por | Quando Consultar |
|-----------|-----------|------------------|
| **GLOSSARIO_CONCEITOS.md** | Todos os scripts | Dúvidas terminológicas |
| **EXEMPLOS_USO_SCRIPTS.md** | Scripts básicos/intermediários | Casos práticos |
| **DOCUMENTACAO_SCRIPTS_AVANCADOS.md** | Scripts avançados | Detalhes técnicos |

### **Scripts por Dependência**
```
run_ingest.py (base para todos)
├── run_query.py
├── list_docs.py
├── search_docs.py
├── analyze_chunks.py
│   └── advanced_metrics.py
│       └── evaluate_rag.py
├── show_vectors.py
│   └── analyze_similarity.py
│       └── evaluate_rag.py
├── list_raw.py
├── analyze_retrieval.py
│   └── evaluate_rag.py
└── experiment.py (usa todos)
```

---

## 🎮 **ATALHOS DE NAVEGAÇÃO**

### **Comandos Rápidos no Terminal**
```bash
# Abrir documentação principal
open INDICE_DOCUMENTACAO.md

# Executar sequência básica
python scripts/run_ingest.py && python scripts/run_query.py

# Análise rápida do sistema
python scripts/list_docs.py && python scripts/analyze_chunks.py

# Avaliação completa
python scripts/advanced_metrics.py && python scripts/evaluate_rag.py
```

### **Atalhos de Navegação por Emoji**
- 🟢 = **Básico** (fundamental para funcionamento)
- 🟡 = **Intermediário** (análise e compreensão)
- 🔴 = **Avançado** (pesquisa e experimentação)
- 📚 = **Documentação** (teoria e conceitos)
- 🧪 = **Experimental** (pesquisa científica)
- 🛠️ = **Prático** (implementação e uso)

---

## 📋 **CHECKLIST DE NAVEGAÇÃO**

### **Antes de Começar**
- [ ] Leu o **[README.md](../../README.md)** para setup
- [ ] Consultou o **[INDICE_DOCUMENTACAO.md](../organization/INDICE_DOCUMENTACAO.md)** para orientação
- [ ] Definiu objetivo de aprendizagem (básico/intermediário/avançado)

### **Durante a Aprendizagem**
- [ ] Seguiu a sequência recomendada para seu nível
- [ ] Consultou **[GLOSSARIO_CONCEITOS.md](../reference/GLOSSARIO_CONCEITOS.md)** quando necessário
- [ ] Experimentou exemplos do **[EXEMPLOS_USO_SCRIPTS.md](EXEMPLOS_USO_SCRIPTS.md)**

### **Para Pesquisa Avançada**
- [ ] Estudou **[DOCUMENTACAO_SCRIPTS_AVANCADOS.md](../reference/DOCUMENTACAO_SCRIPTS_AVANCADOS.md)**
- [ ] Executou scripts em ordem de dependência
- [ ] Documentou descobertas e resultados

---

## 🔄 **LOOPS DE FEEDBACK**

### **Loop de Aprendizagem Básica**
```
📖 Ler teoria → 🛠️ Praticar script → 🔍 Analisar resultado → 📚 Consultar documentação → 🔄 Repetir
```

### **Loop de Pesquisa Avançada**
```
🎯 Definir hipótese → 🧪 Experimentar → 📊 Medir resultados → 📝 Documentar → 🔄 Iterar
```

### **Loop de Resolução de Problemas**
```
🚨 Identificar problema → 🔍 Diagnosticar → 📋 Consultar documentação → 🛠️ Implementar solução → ✅ Validar → 🔄 Monitorar
```

---

## 🎯 **PRÓXIMOS PASSOS SUGERIDOS**

### **Após Nível Básico**
1. Explore **[scripts intermediários](INDICE_DOCUMENTACAO.md#-nível-intermediário---análise-e-compreensão)**
2. Aprofunde com **[DOCUMENTACAO_SCRIPTS_AVANCADOS.md](../reference/DOCUMENTACAO_SCRIPTS_AVANCADOS.md)**
3. Experimente **[casos práticos](EXEMPLOS_USO_SCRIPTS.md)**

### **Após Nível Intermediário**
1. Inicie **[pesquisa avançada](INDICE_DOCUMENTACAO.md#-nível-avançado---pesquisa-e-experimentação)**
2. Conduza **[experimentos científicos](scripts/experiment.py)**
3. Contribua com **documentação adicional**

### **Após Nível Avançado**
1. **Desenvolva novos scripts** baseados no framework existente
2. **Publique pesquisas** baseadas nos experimentos
3. **Contribua para o projeto** com melhorias e extensões

---

*🧭 **Nota de Navegação:** Este guia é atualizado conforme novos documentos e scripts são adicionados. Use-o como referência central para movimentação eficiente pelo sistema educacional.*
