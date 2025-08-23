# config.py
# Configurações centralizadas do projeto RAG

import os
from pathlib import Path

# Diretórios
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
PERSIST_DIR = os.getenv("PERSIST_DIR", "db")

# ChromaDB
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "demo-rag")

# Modelos Ollama
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
EMB_MODEL = os.getenv("EMB_MODEL", "nomic-embed-text")

# Configurações de chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# Configurações de retrieval
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))

# Configurações do LLM
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

# Reset do índice
RESET_CHROMA = os.getenv("RESET_CHROMA", "0") == "1"

# Ollama (opcional)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
