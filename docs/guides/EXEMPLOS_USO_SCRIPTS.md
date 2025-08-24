# 🚀 Exemplos Práticos de Uso dos Scripts

## 📚 Guia Passo-a-Passo para Análise Completa do Sistema RAG

> **Navegação:** 
> - 📚 **[Voltar ao Índice](../organization/INDICE_DOCUMENTACAO.md)** | **[Guia de Navegação](GUIA_NAVEGACAO.md)**
> - 🎯 **[Classificação de Scripts](../organization/CLASSIFICACAO_SCRIPTS.md)** | **[Glossário](../reference/GLOSSARIO_CONCEITOS.md)**

Este documento fornece exemplos práticos e cenários reais de uso dos scripts, organizados por nível de complexidade e com interpretação detalhada dos resultados.

---

## 🎯 **CASOS DE USO POR NÍVEL**

### **🟢 Casos Básicos** - *Primeiros Passos*
- **[Cenário 1: Primeira Ingestão](#-cenário-1-primeira-ingestão)**
- **[Cenário 2: Primeiras Consultas](#-cenário-2-primeiras-consultas)**
- **[Cenário 3: Exploração do Índice](#-cenário-3-exploração-do-índice)**

### **🟡 Casos Intermediários** - *Análise e Otimização*
- **[Cenário 4: Diagnóstico de Qualidade](#-cenário-4-diagnóstico-de-qualidade)**
- **[Cenário 5: Otimização de Chunking](#-cenário-5-otimização-de-chunking)**
- **[Cenário 6: Análise Visual](#-cenário-6-análise-visual)**

### **🔴 Casos Avançados** - *Pesquisa Científica*
- **[Cenário 7: Avaliação Científica](#-cenário-7-avaliação-científica)**
- **[Cenário 8: Experimentação A/B](#-cenário-8-experimentação-ab)**
- **[Cenário 9: Pesquisa Original](#-cenário-9-pesquisa-original)**

---

## 🟢 **CENÁRIOS BÁSICOS**

### 🎯 **Cenário 1: Primeira Ingestão**

**Situação**: Você tem documentos e quer criar seu primeiro índice RAG.

#### **Script:** **[run_ingest.py](scripts/run_ingest.py)** ([Documentação Detalhada](INDICE_DOCUMENTACAO.md#scripts-básicos))

---

## 🔬 Cenário 1: Diagnóstico Inicial do Sistema

### **Situação**: Você acabou de implementar um sistema RAG e quer verificar se está funcionando corretamente.

#### **Passo 1: Verificação Básica da Saúde dos Embeddings**
```bash
# Análise rápida da qualidade dos embeddings
python scripts/advanced_metrics.py --quality --documents
```

**Resultado Esperado**:
```
📊 QUALIDADE DOS EMBEDDINGS:
   Normalização:
     Norma média: 1.000000
     Está normalizado: ✅
   Distribuição:
     Assimetria: 0.085
     Curtose: 0.727
     É normal: ❌
   Correlações:
     Pares altamente correlacionados: 0
     Correlação máxima: 0.234
```

**✅ Interpretação (Sistema Saudável)**:
- Embeddings normalizados ✅ (modelo funciona corretamente)
- Baixa correlação entre dimensões ✅ (sem redundância)
- Distribuição não-normal é aceitável para embeddings

#### **Passo 2: Verificação de Duplicatas**
```bash
# Detectar possíveis duplicatas
python scripts/analyze_similarity.py --duplicates 0.9
```

**Resultado Esperado**:
```
🎯 Encontrados 0 pares similares
```

**✅ Interpretação**: Nenhuma duplicata encontrada - dataset bem curado.

---

## ⚠️ Cenário 2: Identificação de Problemas

### **Situação**: O sistema RAG está retornando resultados inconsistentes.

#### **Diagnóstico Completo**:
```bash
# Análise completa para identificar problemas
python scripts/advanced_metrics.py --all --output diagnostico.json
python scripts/analyze_similarity.py --all
```

**Resultado Problemático**:
```
📊 QUALIDADE DOS EMBEDDINGS:
   Normalização:
     Norma média: 0.847392
     Está normalizado: ❌
   Correlações:
     Pares altamente correlacionados: 45
     Correlação máxima: 0.923

🚨 OUTLIERS (z_score):
   Encontrados: 8
```

**❌ Interpretação (Problemas Identificados)**:
1. **Embeddings não normalizados**: Modelo ou preprocessamento incorreto
2. **Muitas correlações altas**: Redundância no espaço vetorial
3. **Muitos outliers**: Dados inconsistentes ou ruído

#### **Ações Corretivas**:
```bash
# 1. Verificar configuração do modelo de embedding
# 2. Limpar dados de entrada
# 3. Re-treinar ou trocar modelo
# 4. Aplicar normalização manual se necessário
```

---

## 🎯 Cenário 3: Otimização de Performance

### **Situação**: Sistema funciona, mas você quer otimizar para melhor performance.

#### **Experimentação com Diferentes Configurações**:
```bash
# Testar diferentes tamanhos de chunk
python scripts/experiment.py --chunk-sizes 200 500 1000

# Testar diferentes valores de K (documentos recuperados)
python scripts/experiment.py --k-values 3 5 10 15

# Avaliação end-to-end
python scripts/evaluate_rag.py --query "Como funciona o sistema de login?" --output avaliacao.json
```

**Resultado de Experimentação**:
```
📊 EXPERIMENTO - TAMANHOS DE CHUNK:
   Chunk 200: Precisão=0.85, Velocidade=1.2s
   Chunk 500: Precisão=0.91, Velocidade=1.8s  ← ÓTIMO
   Chunk 1000: Precisão=0.87, Velocidade=2.4s

📊 EXPERIMENTO - VALORES DE K:
   K=3: Recall=0.76, Velocidade=0.8s
   K=5: Recall=0.89, Velocidade=1.2s      ← ÓTIMO
   K=10: Recall=0.92, Velocidade=2.1s
```

**📈 Interpretação**:
- **Chunk 500** oferece melhor balanceço precisão/velocidade
- **K=5** é o ponto doce para recall vs velocidade
- Implementar essas configurações otimizadas

---

## 🔍 Cenário 4: Análise de Conteúdo Específico

### **Situação**: Você quer entender como documentos específicos se relacionam.

#### **Análise Visual Detalhada**:
```bash
# Criar visualizações completas
python scripts/analyze_similarity.py --heatmap --clusters 5 --dimensions
```

**Arquivos Gerados**:
- `similarity_heatmap.png`: Matriz visual de similaridades
- `clusters_plot.png`: Agrupamentos temáticos
- `embedding_dimensions.png`: Distribuição por dimensões

#### **Interpretação do Heatmap**:
```
🔥 PADRÕES NO HEATMAP:
   - Bloco quente no canto superior: Documentos sobre "login"
   - Linha diagonal fria: Documento outlier (muito diferente)
   - Distribuição uniforme: Boa diversidade geral
```

#### **Interpretação dos Clusters**:
```
🎭 ANÁLISE DOS CLUSTERS:
   Cluster 1 (40% docs): Documentos técnicos
   Cluster 2 (35% docs): Documentos de usuário
   Cluster 3 (25% docs): Documentos de processo
```

---

## 📊 Cenário 5: Monitoramento Contínuo

### **Situação**: Sistema em produção - você quer monitorar qualidade ao longo do tempo.

#### **Script de Monitoramento Automatizado**:
```bash
# Criar relatório diário completo
python scripts/advanced_metrics.py --all --output "relatorio_$(date +%Y%m%d).json"
python scripts/evaluate_rag.py --batch --output "avaliacao_$(date +%Y%m%d).json"
```

#### **Métricas de Alerta**:
```bash
# Definir thresholds para alertas
NORMALIZAÇÃO_MIN=0.95    # Embeddings devem estar >95% normalizados
OUTLIERS_MAX=5          # Máximo 5% de outliers
DUPLICATAS_MAX=10       # Máximo 10% de duplicatas
TEMPO_RESPOSTA_MAX=3    # Máximo 3 segundos por query
```

#### **Dashboard de Métricas**:
```
📈 DASHBOARD RAG - $(date)
═══════════════════════════════════════

✅ SAÚDE DO SISTEMA:
   Normalização: 99.8% ✅
   Outliers: 2.1% ✅
   Duplicatas: 1.4% ✅
   Tempo Médio: 1.8s ✅

📊 TENDÊNCIAS (7 dias):
   Qualidade: +2.3% ↗️
   Velocidade: +12% ↗️
   Satisfação: 94% ↗️
```

---

## 🛠️ Cenário 6: Debugging de Problemas Específicos

### **Situação**: Usuários relatam que certas queries retornam resultados irrelevantes.

#### **Análise Focada em Retrieval**:
```bash
# Analisar especificamente o sistema de recuperação
python scripts/analyze_retrieval.py --full-analysis

# Testar queries problemáticas específicas
python scripts/evaluate_rag.py --query "login não funciona" --detailed
python scripts/analyze_retrieval.py --queries "login" "autenticação" "acesso"
```

**Resultado de Debugging**:
```
🔍 ANÁLISE DE RETRIEVAL:
   Recall@3: 0.45 ❌ (Muito baixo!)
   Precision@1: 0.67 ⚠️ (Pode melhorar)
   
📋 QUERIES PROBLEMÁTICAS:
   "login não funciona" → Similarity: 0.23 ❌
   "erro de acesso" → Similarity: 0.31 ❌
```

**🔧 Ações de Correção**:
1. **Melhorar chunking**: Queries negativas não casam bem
2. **Expandir vocabulário**: Adicionar sinônimos e variações
3. **Ajustar embeddings**: Modelo pode não capturar negações
4. **Implementar pré-processamento**: Normalizar queries de entrada

---

## 📚 Cenário 7: Análise Educacional Profunda

### **Situação**: Você está estudando como RAG funciona e quer entender os fundamentos matemáticos.

#### **Exploração Matemática Completa**:
```bash
# Análise matemática detalhada
python scripts/advanced_metrics.py --all --entropy --output matematica.json

# Visualizações educacionais
python scripts/analyze_similarity.py --all

# Análise de entropia e diversidade
python scripts/advanced_metrics.py --entropy --documents
```

**Conceitos Explorados**:
```
🎓 CONCEITOS MATEMÁTICOS OBSERVADOS:

📐 ÁLGEBRA LINEAR:
   - Normas L2: ||v|| = √(v₁² + v₂² + ... + vₙ²)
   - Similaridade Cosseno: cos(θ) = (A·B)/(||A||×||B||)
   - Projeções em espaços de alta dimensão

📊 ESTATÍSTICA:
   - Distribuições: Skewness = 0.085 (quase simétrica)
   - Correlações: r_max = 0.234 (baixa interdependência)
   - Outliers: Z-score > 3.0 (detecção de anomalias)

🌀 TEORIA DA INFORMAÇÃO:
   - Entropia: H = -Σ P(x) × log₂ P(x) = 4.632 bits
   - Diversidade: Alta entropia = boa representação
   - Redundância: Baixa correlação = eficiência
```

---

## 🚀 Fluxo de Trabalho Recomendado

### **Para Novos Sistemas RAG**:
```bash
# 1. Verificação inicial
python scripts/advanced_metrics.py --quality
python scripts/analyze_similarity.py --stats --duplicates

# 2. Se problemas encontrados
python scripts/advanced_metrics.py --all --outliers
python scripts/analyze_similarity.py --all

# 3. Otimização
python scripts/experiment.py --full-experiment
python scripts/evaluate_rag.py --batch

# 4. Validação final
python scripts/evaluate_rag.py --query "teste específico"
```

### **Para Sistemas em Produção**:
```bash
# Monitoramento semanal
python scripts/advanced_metrics.py --quality --documents --output weekly_health.json
python scripts/evaluate_rag.py --batch --output weekly_performance.json

# Análise mensal completa
python scripts/advanced_metrics.py --all --output monthly_full_analysis.json
python scripts/analyze_similarity.py --all
python scripts/experiment.py --compare-configs
```

---

## 📋 Checklist de Qualidade

### ✅ **Sistema Saudável**:
- [ ] Embeddings normalizados (`is_normalized: true`)
- [ ] Poucos outliers (<5% dos documentos)
- [ ] Baixas correlações entre dimensões (<10% high_correlation_pairs)
- [ ] Poucas duplicatas (<10% dos pares)
- [ ] Tempo de resposta <3 segundos
- [ ] Recall@5 >0.8 e Precision@3 >0.7
- [ ] Entropia balanceada (>3.0 bits)
- [ ] Clusters bem distribuídos

### ⚠️ **Sinais de Alerta**:
- [ ] `is_normalized: false`
- [ ] Muitos outliers (>10%)
- [ ] Alta correlação entre dimensões (>20%)
- [ ] Muitas duplicatas (>20%)
- [ ] Tempo de resposta >5 segundos
- [ ] Recall baixo (<0.6) ou Precision baixa (<0.5)
- [ ] Entropia muito baixa (<2.0)
- [ ] Um cluster dominante (>80% dos docs)

---

## 🎯 Dicas de Interpretação Rápida

### **Leitura Rápida dos Resultados**:

1. **Primeiro olhe**: `is_normalized` e `outliers`
2. **Segundo verifique**: Número de duplicatas
3. **Terceiro analise**: Tempo de resposta e recall
4. **Quarto observe**: Distribuição dos clusters

### **Ações por Prioridade**:

1. **CRÍTICO**: Corrigir normalização
2. **ALTO**: Remover outliers e duplicatas
3. **MÉDIO**: Otimizar performance (chunk size, K)
4. **BAIXO**: Melhorar distribuição e balanceamento

Este guia prático permite que você use os scripts de forma efetiva para diagnosticar, otimizar e monitorar seu sistema RAG! 🚀
