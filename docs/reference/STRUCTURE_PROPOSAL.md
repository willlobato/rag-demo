# Proposta de Reestruturação do Projeto

## Estrutura Atual (Problemática)
```
rag-demo/
├── ingest.py           # Scripts na raiz
├── rag_ollama.py
├── list_docs.py
├── list_raw.py
├── search_docs.py
├── requirements.txt
├── data/
└── db/
```

## Estrutura Recomendada
```
rag-demo/
├── src/
│   └── rag_demo/              # Pacote principal
│       ├── __init__.py
│       ├── core/              # Funcionalidades principais
│       │   ├── __init__.py
│       │   ├── ingest.py      # Lógica de ingestão
│       │   ├── retrieval.py   # Lógica de busca
│       │   └── rag.py         # Sistema RAG principal
│       ├── utils/             # Utilitários
│       │   ├── __init__.py
│       │   ├── document_loader.py
│       │   └── embeddings.py
│       └── config.py          # Configurações centralizadas
├── scripts/                   # Scripts executáveis
│   ├── ingest_documents.py
│   ├── query_rag.py
│   ├── list_documents.py
│   └── search_documents.py
├── tests/                     # Testes unitários
│   ├── __init__.py
│   ├── test_core/
│   └── test_utils/
├── data/                      # Dados de entrada
├── db/                        # Banco vetorial
├── docs/                      # Documentação
├── .gitignore                 # Arquivos a ignorar
├── .env.example               # Exemplo de variáveis
├── pyproject.toml             # Configuração moderna Python
├── requirements.txt           # Dependências
└── README.md
```

## Benefícios da Nova Estrutura

### 1. **Separação de Responsabilidades**
- `src/rag_demo/core/` - Lógica principal
- `src/rag_demo/utils/` - Funções auxiliares  
- `scripts/` - Scripts executáveis
- `tests/` - Testes organizados

### 2. **Facilita Importações**
```python
# Antes (problemático)
from ingest import load_txt_md  # Pode conflitar

# Depois (limpo)
from rag_demo.core.ingest import load_txt_md
from rag_demo.utils.document_loader import PDFLoader
```

### 3. **Configuração Centralizada**
```python
# src/rag_demo/config.py
from pathlib import Path
import os

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
PERSIST_DIR = os.getenv("PERSIST_DIR", "db") 
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "demo-rag")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
EMB_MODEL = os.getenv("EMB_MODEL", "nomic-embed-text")
```

### 4. **Instalação como Pacote**
```bash
# Instalar em modo desenvolvimento
pip install -e .

# Usar em qualquer lugar
python -c "from rag_demo.core.rag import build_rag_chain"
```

## Arquivos Importantes Faltando

### .gitignore
```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
.venv/
.env
db/
*.log
.DS_Store
.vscode/
```

### pyproject.toml (Padrão moderno)
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "rag-demo"
version = "0.1.0"
description = "RAG system with Ollama and ChromaDB"
authors = [{name = "Seu Nome", email = "seu@email.com"}]
dependencies = [
    "langchain>=0.2.11",
    "langchain-core>=0.2.35",
    "langchain-community>=0.2.10",
    "langchain-ollama>=0.1.0",
    "langchain-chroma>=0.1.2",
    "chromadb>=0.5.3",
    "tiktoken>=0.7.0",
    "pypdf>=4.2.0",
]

[project.scripts]
rag-ingest = "rag_demo.scripts.ingest:main"
rag-query = "rag_demo.scripts.query:main"
```

## Migração Gradual

1. **Criar estrutura de pastas**
2. **Mover arquivos por etapas**
3. **Atualizar imports**
4. **Adicionar __init__.py**
5. **Testar funcionalidades**

## Para Projetos Pequenos (Alternativa Simples)

Se quiser manter simples mas melhorar:

```
rag-demo/
├── rag_demo/              # Pasta principal
│   ├── __init__.py
│   ├── ingest.py
│   ├── rag.py
│   ├── utils.py
│   └── config.py
├── scripts/               # Scripts CLI
│   ├── run_ingest.py
│   ├── run_query.py
│   └── list_docs.py
├── data/
├── .gitignore
├── requirements.txt
└── README.md
```

## Conclusão

A estrutura atual **funciona para aprendizado**, mas para um projeto mais sério, recomendo a reestruturação. Começaria pela versão simples e evoluiria conforme necessário.
