# __init__.py
# Pacote rag_demo - Sistema RAG com Ollama e ChromaDB

from .config import *
from .ingest import ingest_documents
from .rag import query_rag, search_similar_docs, list_documents

__version__ = "0.1.0"
__author__ = "Will Lobato"
__description__ = "Sistema RAG com Ollama e ChromaDB"

# Exportar funções principais
__all__ = [
    "ingest_documents",
    "query_rag", 
    "search_similar_docs",
    "list_documents"
]
