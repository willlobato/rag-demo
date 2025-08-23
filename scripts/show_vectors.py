#!/usr/bin/env python3
# show_vectors.py
# Script para mostrar os valores reais dos vetores

import sys
from pathlib import Path
import numpy as np

# Adicionar o diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_demo.rag import build_or_load_vectorstore
from rag_demo.config import EMB_MODEL
from langchain_ollama import OllamaEmbeddings


def main():
    """Mostra os vetores reais dos embeddings."""
    if len(sys.argv) < 2:
        print("Uso: python show_vectors.py \"sua consulta\" [mostrar_vetores]")
        return
    
    query = sys.argv[1]
    show_vectors = len(sys.argv) > 2 and sys.argv[2].lower() == "true"
    
    try:
        vector_store = build_or_load_vectorstore()
        embeddings = OllamaEmbeddings(model=EMB_MODEL)
        
        # Primeiro, vamos obter o embedding da query
        query_embedding = embeddings.embed_query(query)
        
        print(f"Query: {query}")
        print(f"Embeddings da query: {len(query_embedding)} dimensões")
        
        if show_vectors:
            print(f"Vetor da query: {np.array(query_embedding)[:10]}... (primeiros 10 valores)")
        
        # Busca documentos similares usando o vector store
        results = vector_store.similarity_search_with_score(query, k=3)
        
        print(f"\n=== Documentos Similares ===")
        for i, (doc, score) in enumerate(results, 1):
            print(f"\nDoc {i} (score: {score:.4f}):")
            print(f"  Conteúdo: {doc.page_content[:100]}...")
            print(f"  Fonte: {doc.metadata.get('source', 'N/A')}")
        
        # Busca todos os documentos com seus vetores usando ChromaDB diretamente
        all_docs = vector_store._collection.get(
            include=["embeddings", "documents", "metadatas"]
        )
        
        print(f"\n=== Documentos Armazenados ({len(all_docs['documents'])}) ===")
        for i, (doc, metadata, embedding) in enumerate(zip(
            all_docs['documents'], 
            all_docs['metadatas'], 
            all_docs['embeddings']
        )):
            print(f"\nDoc {i+1}:")
            print(f"  Conteúdo: {doc[:100]}...")
            print(f"  Fonte: {metadata.get('source', 'N/A')}")
            print(f"  Vetor: {len(embedding)} dimensões")
            
            if show_vectors:
                vector_array = np.array(embedding)
                print(f"  Primeiros 10 valores: {vector_array[:10]}")
                print(f"  Últimos 10 valores: {vector_array[-10:]}")
                print(f"  Norma L2: {np.linalg.norm(vector_array):.4f}")
    
    except Exception as e:
        print(f"[ERROR] Erro ao acessar vetores: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
