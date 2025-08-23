# RAG Demo - Recuperação e Geração Aumentada com Ollama

Um projeto de demonstração de RAG (Retrieval-Augmented Generation) usando Ollama para modelos locais, ChromaDB para armazenamento vetorial e LangChain para orquestração.

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

### 2. Python 3.8+

Certifique-se de ter Python 3.8 ou superior instalado.

## 🚀 Instalação

1. **Clone ou baixe o projeto**

2. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

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
│   └── search_docs.py     # Script para busca semântica
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
```

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
