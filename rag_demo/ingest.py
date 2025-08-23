# ingest.py
# Módulo para ingestão e indexação de documentos

import os
import shutil
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from .config import (
    DATA_DIR, PERSIST_DIR, COLLECTION_NAME, EMB_MODEL, 
    CHUNK_SIZE, CHUNK_OVERLAP, RESET_CHROMA
)
from .utils import load_txt_md, load_pdfs, get_example_documents


def reset_index_if_needed():
    """Reseta o índice ChromaDB se a variável RESET_CHROMA estiver ativa."""
    if RESET_CHROMA and Path(PERSIST_DIR).exists():
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        print(f"[INFO] Reset de índice: removido '{PERSIST_DIR}/'.")


def load_all_documents():
    """Carrega todos os documentos do diretório de dados."""
    if not DATA_DIR.exists():
        print("[INFO] Pasta 'data/' não encontrada. Criando vazia.")
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_docs = load_txt_md(DATA_DIR) + load_pdfs(DATA_DIR)

    if not raw_docs:
        raw_docs = get_example_documents()
        print("[INFO] Sem arquivos em data/. Usando documentos de exemplo.")

    return raw_docs


def create_chunks(documents):
    """Divide documentos em chunks usando RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"[INFO] Chunks gerados: {len(chunks)}")
    return chunks


def create_vector_store(chunks):
    """Cria o armazenamento vetorial com ChromaDB."""
    embeddings = OllamaEmbeddings(model=EMB_MODEL)

    # Chroma 0.4+ persiste automaticamente
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )

    print(f"[OK] Índice atualizado em '{PERSIST_DIR}/' (coleção: {COLLECTION_NAME})")


def ingest_documents():
    """Função principal para ingestão de documentos."""
    print("[INFO] Iniciando processo de ingestão...")
    
    # Reset opcional do índice
    reset_index_if_needed()
    
    # Carregar documentos
    documents = load_all_documents()
    
    # Criar chunks
    chunks = create_chunks(documents)
    
    # Criar store vetorial
    create_vector_store(chunks)
    
    print("[INFO] Processo de ingestão concluído!")
