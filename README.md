# RAG Demo - Recuperação e Geração Aumentada com Ollama

Um projeto de demonstração de RAG (Retrieval-Augmented Generation) usando Ollama para modelos locais, ChromaDB para armazenamento vetorial e LangChain para orquestração.

> 📚 **[Ver Tutorial Completo sobre RAG](TUTORIAL_RAG.md)** - Guia aprofundado sobre conceitos, arquitetura e todos os componentes técnicos.

## 📋 Sobre o Projeto

Este projeto implementa um sistema RAG completo que:

- **Processa documentos** (TXT, MD, PDF) e os divide em chunks
- **Gera embeddings** usando o modelo `nomic-embed-text` do Ollama
- **Armazena vetores** no ChromaDB com persistência automática
- **Responde perguntas** usando o modelo `llama3` com contexto recuperado

## 🛠️ Pré-requisitos

### 1. Ollama

Instale o Ollama e baixe os modelos necessários:

```bash
# Instalar Ollama (macOS)
brew install ollama

# Iniciar o serviço
brew services start ollama

# Baixar os modelos
ollama pull llama3
ollama pull nomic-embed-text
```

### 2. Python 3.10+

Certifique-se de ter Python 3.10 ou superior instalado.

## 🚀 Instalação

1. **Clone ou baixe o projeto**

2. **Configure o ambiente virtual (recomendado):**
   ```bash
   # Criar ambiente virtual
   python -m venv venv
   
   # Ativar ambiente virtual
   # No macOS/Linux:
   source venv/bin/activate
   # No Windows:
   # venv\Scripts\activate
   
   # Verificar se está ativo (deve mostrar (venv) no prompt)
   which python
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Para desativar o ambiente virtual (quando terminar):**
   ```bash
   deactivate
   ```

> **💡 Por que usar ambiente virtual?**
> - **Isolamento**: Evita conflitos entre dependências de diferentes projetos
> - **Versões específicas**: Cada projeto mantém suas próprias versões de bibliotecas
> - **Limpeza**: Não afeta o Python global do sistema
> - **Reprodutibilidade**: Garante funcionamento consistente em outras máquinas

## 📁 Estrutura do Projeto

```
rag-demo/
├── rag_demo/              # Pacote principal
│   ├── __init__.py        # Inicialização do pacote
│   ├── config.py          # Configurações centralizadas
│   ├── ingest.py          # Lógica de ingestão de documentos
│   ├── rag.py             # Sistema RAG principal
│   └── utils.py           # Funções utilitárias
├── scripts/               # Scripts executáveis
│   ├── run_ingest.py      # Script para ingestão
│   ├── run_query.py       # Script para consultas RAG
│   ├── list_docs.py       # Script para listar documentos
│   ├── search_docs.py     # Script para busca semântica
│   ├── show_vectors.py    # Script para visualizar vetores
│   ├── analyze_chunks.py  # Script para analisar chunks e overlaps
│   ├── list_raw.py        # Script simples para listar chunks + embeddings
│   ├── evaluate_rag.py    # Script para avaliação de qualidade RAG
│   ├── analyze_similarity.py # Script para análise de similaridade
│   ├── experiment.py      # Script para experimentação com configurações
│   ├── analyze_retrieval.py  # Script para análise de recuperação
│   └── advanced_metrics.py   # Script para métricas avançadas
├── requirements.txt       # Dependências Python
├── .gitignore            # Arquivos a ignorar no Git
├── .env.example          # Template de variáveis de ambiente
├── data/                 # Documentos de entrada
│   └── guia.txt          # Exemplo de documento
└── db/                   # Banco vetorial ChromaDB (criado automaticamente)
```

## 🎯 Como Usar

### 1. Indexar Documentos

```bash
# Processar documentos da pasta data/
python scripts/run_ingest.py

# Para resetar o índice completamente
RESET_CHROMA=1 python scripts/run_ingest.py
```

O script processa automaticamente:
- Arquivos `.txt` e `.md` da pasta `data/`
- Arquivos `.pdf` da pasta `data/`
- Se não houver arquivos, usa documentos de exemplo

### 2. Fazer Consultas RAG

```bash
# Pergunta padrão
python scripts/run_query.py

# Pergunta customizada
python scripts/run_query.py "Como otimizar o processo de login?"
python scripts/run_query.py "O que é RAG e como funciona?"
```

### 3. Explorar o Índice

```bash
# Listar todos os documentos indexados
python scripts/list_docs.py

# Busca semântica direta (sem LLM)
python scripts/search_docs.py "otimização de login" 3

# Visualizar vetores e embeddings
python scripts/show_vectors.py "login"
python scripts/show_vectors.py "login" true  # Mostra valores reais dos vetores

# Listagem simples de chunks + embeddings
python scripts/list_raw.py

# Analisar chunks e overlaps detalhadamente
python scripts/analyze_chunks.py
python scripts/analyze_chunks.py --full  # Conteúdo completo dos chunks
```

### 4. Analisar Vetores

O script `show_vectors.py` permite visualizar os embeddings em detalhes:

```bash
# Informações básicas dos vetores
python scripts/show_vectors.py "cache distribuído"

# Valores numéricos completos dos vetores
python scripts/show_vectors.py "cache distribuído" true
```

**O que você verá:**
- **Dimensões dos vetores** (768 para nomic-embed-text)
- **Scores de similaridade** entre query e documentos
- **Valores reais dos embeddings** (arrays NumPy)
- **Norma L2** dos vetores (normalizados para 1.0)
- **Primeiros e últimos 10 valores** de cada vetor

### 5. Listagem Simples de Chunks

Para uma visualização rápida e direta de todos os chunks:

```bash
# Lista todos os chunks com texto completo + primeiros 10 embeddings
python scripts/list_raw.py
```

**Ideal para:**
- Verificar rapidamente o que está indexado
- Ver o texto completo de cada chunk
- Conferir os primeiros valores dos embeddings
- Uso simples sem parâmetros

### 6. Análise de Chunks e Overlaps

Para entender como os documentos foram divididos:

```bash
# Análise completa de chunking
python scripts/analyze_chunks.py

# Com conteúdo completo dos chunks
python scripts/analyze_chunks.py --full
```

**Mostra:**
- Estatísticas de chunking (tamanhos, médias)
- Overlaps entre chunks consecutivos
- Conteúdo exato das sobreposições
- Chunks detalhados por documento

## 🔬 Scripts Avançados para Análise

### 7. Avaliação de Qualidade RAG

```bash
# Avaliação com dataset de exemplo
python scripts/evaluate_rag.py --sample

# Avaliação de pergunta específica
python scripts/evaluate_rag.py --question "Como foi otimizado o login?"

# Avaliação com dataset personalizado
python scripts/evaluate_rag.py --dataset questions.json --output results.json
```

**Métricas calculadas:**
- Similaridade com respostas esperadas
- Relevância das respostas
- Fidelidade ao contexto (faithfulness)
- Tempo de resposta

### 8. Análise de Similaridade

```bash
# Análise completa de similaridade
python scripts/analyze_similarity.py --all

# Apenas heatmap de similaridade
python scripts/analyze_similarity.py --heatmap

# Detectar chunks duplicados (threshold 90%)
python scripts/analyze_similarity.py --duplicates 0.9

# Clustering de documentos
python scripts/analyze_similarity.py --clusters 5
```

**Funcionalidades:**
- Heatmap de similaridade entre chunks
- Detecção de duplicatas
- Clustering de documentos por temas
- Visualização PCA 2D

### 9. Experimentação com Configurações

```bash
# Comparar diferentes tamanhos de chunk
python scripts/experiment.py --compare-chunks --chunk-sizes 250,500,1000

# Comparar valores de K para recuperação
python scripts/experiment.py --compare-k --retrieval-k 3,5,7,10

# Executar todos os experimentos
python scripts/experiment.py --all

# Criar arquivo de perguntas de exemplo
python scripts/experiment.py --create-questions
```

**Permite testar:**
- Diferentes chunk sizes e overlaps
- Valores de K para recuperação
- Modelos de embedding
- Batch testing com múltiplas queries

### 10. Análise de Recuperação

```bash
# Análise completa de recuperação
python scripts/analyze_retrieval.py --all

# Analisar query específica
python scripts/analyze_retrieval.py --query "otimização de performance"

# Encontrar queries problemáticas
python scripts/analyze_retrieval.py --problems --threshold 0.8

# Analisar popularidade dos chunks
python scripts/analyze_retrieval.py --popularity
```

**Métricas incluídas:**
- Recall@K e Precision@K
- Distribuição de scores de similaridade
- Chunks mais/menos recuperados
- Queries com baixa qualidade de recuperação

### 11. Métricas Avançadas

```bash
# Análise matemática completa
python scripts/advanced_metrics.py --all

# Apenas qualidade dos embeddings
python scripts/advanced_metrics.py --quality

# Detectar outliers
python scripts/advanced_metrics.py --outliers z_score

# Análise de entropia
python scripts/advanced_metrics.py --entropy --output metrics.json
```

**Análises matemáticas:**
- Qualidade e normalização dos embeddings
- Distribuições de distância (Euclidiana, Cosseno, Manhattan)
- Detecção de outliers
- Entropia e correlações entre dimensões
- Características estatísticas dos documentos

## ⚙️ Configuração

### Modelos

Você pode alterar os modelos editando as constantes em `rag_demo/config.py`:

```python
# Em rag_demo/config.py
LLM_MODEL = "llama3"          # Modelo para geração de texto
EMB_MODEL = "nomic-embed-text" # Modelo para embeddings
```

### Parâmetros de Chunking

```python
# Em rag_demo/config.py
CHUNK_SIZE = 500      # Tamanho máximo do chunk
CHUNK_OVERLAP = 80    # Sobreposição entre chunks
```

### Recuperação

```python
# Em rag_demo/config.py
RETRIEVAL_K = 4  # Número de chunks recuperados
```

### Variáveis de Ambiente

Copie `.env.example` para `.env` e ajuste as configurações:

```bash
cp .env.example .env
# Edite o arquivo .env conforme necessário
```

## 📚 Funcionalidades Principais

### `rag_demo/ingest.py`
- Carrega documentos TXT, MD e PDF
- Divide em chunks com sobreposição
- Gera embeddings e salva no ChromaDB
- Suporte para reset do índice

### `rag_demo/rag.py`
- Sistema RAG completo
- Recupera contexto relevante
- Gera respostas com modelo Ollama
- Inclui fontes nas respostas

### `rag_demo/config.py`
- Configurações centralizadas
- Suporte a variáveis de ambiente
- Parâmetros configuráveis

### Scripts Executáveis
- **`scripts/run_ingest.py`**: Executa a ingestão de documentos
- **`scripts/run_query.py`**: Faz consultas RAG
- **`scripts/list_docs.py`**: Lista documentos indexados
- **`scripts/search_docs.py`**: Busca semântica pura
- **`scripts/show_vectors.py`**: Visualiza embeddings e valores dos vetores
- **`scripts/analyze_chunks.py`**: Analisa chunks e overlaps detalhadamente
- **`scripts/list_raw.py`**: Listagem simples de chunks com embeddings
- **`scripts/evaluate_rag.py`**: Avaliação de qualidade do sistema RAG
- **`scripts/analyze_similarity.py`**: Análise de similaridade entre chunks
- **`scripts/experiment.py`**: Experimentação com diferentes configurações
- **`scripts/analyze_retrieval.py`**: Análise detalhada da qualidade de recuperação
- **`scripts/advanced_metrics.py`**: Métricas avançadas e insights matemáticos

## 🔧 Troubleshooting

### Ollama não está rodando
```bash
# Verificar se está ativo
ollama list

# Iniciar manualmente
ollama serve
```

### Modelos não encontrados
```bash
# Baixar modelos necessários
ollama pull llama3
ollama pull nomic-embed-text
```

### Erro de dependências
```bash
# Reinstalar dependências
pip install -r requirements.txt --upgrade
```

### ChromaDB corrompido
```bash
# Resetar completamente o índice
RESET_CHROMA=1 python scripts/run_ingest.py
```

## 📄 Exemplo de Uso

```bash
# 1. Indexar documentos
python scripts/run_ingest.py

# 2. Fazer uma pergunta
python scripts/run_query.py "Como foi otimizado o processo de login?"

# Saída esperada:
# [Q] Como foi otimizado o processo de login?
# 
# O processo de login foi otimizado através de duas principais estratégias:
# 1. Implementação de cache distribuído com Infinispan
# 2. Utilização de paralelismo nas chamadas de API
# 
# Essas otimizações resultaram numa redução significativa do tempo médio
# de login, passando de 4 segundos para 1,2 segundos.
#
# Fontes: data/guia.txt
```

## 🤝 Contribuição

Este é um projeto de demonstração educacional. Sinta-se à vontade para:

- Adicionar novos tipos de documentos
- Melhorar o sistema de prompt
- Implementar interfaces web
- Adicionar métricas de avaliação

## 📝 Licença

Projeto educacional de código aberto.
