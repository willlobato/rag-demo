# 🗺️ Mapa Visual da Documentação RAG Demo

> **Representação visual das interconexões entre todos os documentos e scripts**
> 
> Este mapa facilita a compreensão da estrutura organizacional e dos fluxos de navegação do sistema educacional.

---

## 🎯 **ESTRUTURA HIERÁRQUICA DA DOCUMENTAÇÃO**

```
📚 RAG DEMO - SISTEMA EDUCACIONAL
│
├── 🏗️ CORE PACKAGE (rag_demo/)
│   ├── __init__.py ......................... (Arquitetura de pacote)
│   ├── config.py ........................... (Configuração avançada)
│   ├── ingest.py ........................... (Pipeline de ingestão)
│   ├── rag.py .............................. (Arquitetura RAG)
│   └── utils.py ............................ (Utilitários educacionais)
│
├── 📖 DOCUMENTAÇÃO PRINCIPAL
│   ├── README.md ........................... (Entrada principal)
│   ├── INDICE_DOCUMENTACAO.md .............. (Navegação estruturada)
│   └── GUIA_NAVEGACAO.md ................... (Mapa de navegação)
│
├── 🎓 FUNDAMENTAÇÃO TEÓRICA
│   ├── TUTORIAL_RAG.md ..................... (Base conceitual)
│   └── GLOSSARIO_CONCEITOS.md .............. (Dicionário técnico)
│
├── 📊 ORGANIZAÇÃO E CLASSIFICAÇÃO
│   ├── CLASSIFICACAO_SCRIPTS.md ............ (Níveis de complexidade)
│   ├── EXEMPLOS_USO_SCRIPTS.md ............. (Casos práticos)
│   └── DOCUMENTACAO_SCRIPTS_AVANCADOS.md ... (Detalhes técnicos)
│
├── 🎯 SISTEMA EDUCACIONAL
│   ├── RESUMO_SISTEMA_EDUCACIONAL.md ....... (Visão geral educacional)
│   └── [Este arquivo: MAPA_VISUAL.md] ...... (Mapa visual)
│
└── 💻 SCRIPTS PRÁTICOS
    ├── 🟢 BÁSICOS (4 scripts)
    ├── 🟡 INTERMEDIÁRIOS (3 scripts)
    └── 🔴 AVANÇADOS (5 scripts)
```

---

## 🔄 **FLUXOS DE NAVEGAÇÃO VISUAL**

### **🎯 Fluxo para Novos Usuários**
```
START → README.md
           ↓
    INDICE_DOCUMENTACAO.md
           ↓
    TUTORIAL_RAG.md (teoria)
           ↓
    CLASSIFICACAO_SCRIPTS.md (níveis)
           ↓
    🟢 Scripts Básicos → 🟡 Scripts Intermediários → 🔴 Scripts Avançados
```

### **🔬 Fluxo para Pesquisadores**
```
START → DOCUMENTACAO_SCRIPTS_AVANCADOS.md
           ↓
    CLASSIFICACAO_SCRIPTS.md (nível avançado)
           ↓
    🔴 Scripts Avançados
           ↓
    EXEMPLOS_USO_SCRIPTS.md (casos científicos)
```

### **🛠️ Fluxo para Resolução de Problemas**
```
PROBLEMA → GUIA_NAVEGACAO.md (diagnóstico)
              ↓
       Script Específico (solução)
              ↓
       EXEMPLOS_USO_SCRIPTS.md (casos similares)
              ↓
       GLOSSARIO_CONCEITOS.md (conceitos)
```

---

## 📊 **MATRIZ DE INTERCONEXÕES**

### **Documentos de Entrada (Pontos de Acesso)**
| Documento | Para Quem | Leva Para |
|-----------|-----------|-----------|
| **README.md** | Todos | INDICE_DOCUMENTACAO.md, Setup |
| **INDICE_DOCUMENTACAO.md** | Organizadores | Qualquer documento |
| **TUTORIAL_RAG.md** | Aprendizes | Scripts práticos |
| **GUIA_NAVEGACAO.md** | Navegadores | Soluções específicas |

### **Documentos de Apoio (Referência)**
| Documento | Usado Por | Consultado Quando |
|-----------|-----------|-------------------|
| **GLOSSARIO_CONCEITOS.md** | Todos | Dúvidas terminológicas |
| **CLASSIFICACAO_SCRIPTS.md** | Estudantes | Escolha de nível |
| **EXEMPLOS_USO_SCRIPTS.md** | Praticantes | Casos reais |
| **DOCUMENTACAO_SCRIPTS_AVANCADOS.md** | Pesquisadores | Detalhes técnicos |

### **Scripts por Dependência**
```
run_ingest.py (OBRIGATÓRIO PRIMEIRO)
├── run_query.py (teste básico)
├── list_docs.py (exploração)
├── search_docs.py (busca)
│
├── analyze_chunks.py (análise)
│   └── advanced_metrics.py (métricas)
│       └── evaluate_rag.py (avaliação)
│
├── show_vectors.py (visualização)
│   └── analyze_similarity.py (similaridade)
│       └── evaluate_rag.py (avaliação)
│
├── list_raw.py (dados brutos)
├── analyze_retrieval.py (recuperação)
│   └── evaluate_rag.py (avaliação)
│
└── experiment.py (usa TODOS os anteriores)
```

---

## 🎮 **INTERFACES DE NAVEGAÇÃO**

### **🏠 Portal Principal**
```
INDICE_DOCUMENTACAO.md
├── Por Nível de Complexidade
│   ├── 🟢 Básico (4 scripts)
│   ├── 🟡 Intermediário (3 scripts)
│   └── 🔴 Avançado (5 scripts)
├── Por Objetivo de Aprendizagem
│   ├── 📚 Fundamentação
│   ├── 🛠️ Prática
│   └── 🔬 Pesquisa
└── Por Fluxo de Trabalho
    ├── Setup → Básico → Intermediário → Avançado
    └── Problema → Diagnóstico → Solução → Validação
```

### **🧭 Portal de Navegação**
```
GUIA_NAVEGACAO.md
├── Por Tipo de Usuário
│   ├── 👨‍🎓 Estudante
│   ├── 👨‍💻 Desenvolvedor
│   ├── 👨‍🔬 Pesquisador
│   └── 👨‍💼 Gestor
├── Por Problema Específico
│   ├── "Resultados irrelevantes"
│   ├── "Chunks inadequados"
│   └── "Performance baixa"
└── Por Objetivo de Pesquisa
    ├── Otimização
    ├── Comparação
    └── Validação
```

---

## 🎯 **CÓDIGOS DE COR E SÍMBOLOS**

### **Por Nível de Complexidade**
- 🟢 **Verde**: Básico (fundamental, fácil)
- 🟡 **Amarelo**: Intermediário (análise, moderado)
- 🔴 **Vermelho**: Avançado (pesquisa, difícil)

### **Por Tipo de Conteúdo**
- 📚 **Documentação**: Teoria e conceitos
- 💻 **Scripts**: Código executável
- 🛠️ **Prático**: Implementação e uso
- 🔬 **Científico**: Pesquisa e experimentação
- 🎓 **Educacional**: Aprendizagem e ensino

### **Por Status de Pré-requisito**
- ✅ **Obrigatório**: Deve ser executado primeiro
- ⚠️ **Recomendado**: Melhor fazer antes
- 💡 **Opcional**: Pode ajudar mas não é necessário

---

## 📋 **CHECKLIST DE NAVEGAÇÃO**

### **Para Iniciantes (🟢)**
- [ ] Leu **README.md** para setup
- [ ] Consultou **INDICE_DOCUMENTACAO.md** para orientação
- [ ] Estudou **TUTORIAL_RAG.md** para base teórica
- [ ] Verificou **CLASSIFICACAO_SCRIPTS.md** para nível básico
- [ ] Executou scripts básicos em ordem

### **Para Analistas (🟡)**
- [ ] Completou nível básico
- [ ] Consultou **EXEMPLOS_USO_SCRIPTS.md** para casos práticos
- [ ] Domina conceitos do **GLOSSARIO_CONCEITOS.md**
- [ ] Executa scripts intermediários independentemente

### **Para Pesquisadores (🔴)**
- [ ] Completou níveis anteriores
- [ ] Estudou **DOCUMENTACAO_SCRIPTS_AVANCADOS.md**
- [ ] Capaz de conduzir experimentos científicos
- [ ] Contribui com documentação adicional

---

## 🚀 **PRÓXIMOS PASSOS POR PERFIL**

### **🎓 Estudante/Aprendiz**
1. **Começa:** README.md → TUTORIAL_RAG.md
2. **Pratica:** Scripts 🟢 → Scripts 🟡
3. **Avança:** EXEMPLOS_USO_SCRIPTS.md

### **👨‍💻 Desenvolvedor/Engenheiro**
1. **Começa:** README.md → CLASSIFICACAO_SCRIPTS.md
2. **Implementa:** Scripts 🟢 e 🟡
3. **Otimiza:** Scripts 🔴 selecionados

### **👨‍🔬 Pesquisador/Cientista**
1. **Começa:** DOCUMENTACAO_SCRIPTS_AVANCADOS.md
2. **Experimenta:** Scripts 🔴 completos
3. **Inova:** Contribuições originais

### **👨‍💼 Gestor/Líder**
1. **Começa:** README.md → RESUMO_SISTEMA_EDUCACIONAL.md
2. **Avalia:** Scripts 🟢 básicos
3. **Estratégia:** Visão geral dos 🔴 avançados

---

## 🔗 **LINKS RÁPIDOS POR CATEGORIA**

### **📚 Documentação Essencial**
- **[README.md](../../README.md)** - Porta de entrada
- **[INDICE_DOCUMENTACAO.md](INDICE_DOCUMENTACAO.md)** - Navegação central
- **[TUTORIAL_RAG.md](../guides/TUTORIAL_RAG.md)** - Base teórica

### **🎯 Organização e Estrutura**
- **[CLASSIFICACAO_SCRIPTS.md](CLASSIFICACAO_SCRIPTS.md)** - Níveis de complexidade
- **[GUIA_NAVEGACAO.md](../guides/GUIA_NAVEGACAO.md)** - Fluxos de navegação
- **[EXEMPLOS_USO_SCRIPTS.md](../guides/EXEMPLOS_USO_SCRIPTS.md)** - Casos práticos

### **💻 Scripts por Nível**
- **[🟢 Básicos](CLASSIFICACAO_SCRIPTS.md#-nível-básico---fundamentos-essenciais)** - run_ingest.py, run_query.py, list_docs.py, search_docs.py
- **[🟡 Intermediários](CLASSIFICACAO_SCRIPTS.md#-nível-intermediário---análise-e-compreensão)** - analyze_chunks.py, show_vectors.py, list_raw.py
- **[🔴 Avançados](CLASSIFICACAO_SCRIPTS.md#-nível-avançado---pesquisa-e-experimentação)** - advanced_metrics.py, analyze_similarity.py, analyze_retrieval.py, evaluate_rag.py, experiment.py

### **🔧 Suporte e Referência**
- **[GLOSSARIO_CONCEITOS.md](../reference/GLOSSARIO_CONCEITOS.md)** - Dicionário técnico
- **[DOCUMENTACAO_SCRIPTS_AVANCADOS.md](../reference/DOCUMENTACAO_SCRIPTS_AVANCADOS.md)** - Detalhes técnicos

---

*🗺️ **Nota:** Este mapa visual é atualizado conforme o sistema evolui. Use-o como referência central para navegação eficiente e compreensão da estrutura organizacional.*
