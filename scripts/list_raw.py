#!/usr/bin/env python3
"""
📄 LIST RAW - VISUALIZAÇÃO EDUCACIONAL DE DADOS BRUTOS DO VECTOR STORE

Este script oferece acesso direto aos dados brutos armazenados no ChromaDB,
proporcionando uma compreensão profunda de como as informações são
estruturadas e organizadas internamente no sistema RAG.

🎯 FUNCIONALIDADES EDUCACIONAIS:

🔍 EXPLORAÇÃO DE DADOS BRUTOS:
- Acesso direto aos dados do ChromaDB
- Visualização de embeddings, documentos e metadados
- Estrutura interna de armazenamento
- Relacionamentos entre diferentes componentes

📊 ANÁLISE ESTRUTURAL:
- Como embeddings são armazenados (arrays numéricos)
- Organização de metadados (dicionários estruturados)
- Textos originais vs textos processados
- IDs únicos e sistemas de referência

🗃️ FORMATO DE DADOS:
- Collections: Agrupamentos lógicos
- Documents: Arrays de strings (chunks de texto)
- Embeddings: Arrays de floats (representações vetoriais)
- Metadatas: Arrays de dicionários (informações auxiliares)
- IDs: Identificadores únicos para cada chunk

📖 CONCEITOS DEMONSTRADOS:

DATABASE STRUCTURE:
- Schema flexível do ChromaDB
- Separação entre dados e índices
- Eficiência de armazenamento
- Estratégias de compressão

DATA RELATIONSHIPS:
- 1:1 entre documento, embedding e metadata
- Preservação de ordem e consistência
- Integridade referencial
- Versionamento de dados

STORAGE EFFICIENCY:
- Otimizações de espaço em disco
- Índices para busca rápida
- Compressão de embeddings
- Metadados estruturados vs flexíveis

🔬 INSIGHTS TÉCNICOS:

RAW DATA FORMAT:
- Embeddings como arrays numpy
- Metadados como JSON estruturado
- Textos em encoding UTF-8
- IDs como strings ou UUIDs

PERFORMANCE IMPLICATIONS:
- Tamanho dos embeddings vs velocidade
- Quantidade de metadados vs overhead
- Estratégias de indexação
- Trade-offs memória vs disco

DEBUGGING CAPABILITIES:
- Validação de integridade dos dados
- Detecção de corrupção
- Análise de inconsistências
- Verificação de duplicatas

🚀 USO EDUCACIONAL:
Este script é essencial para compreender "under the hood" como
um vector store funciona, revelando os detalhes de implementação
que ficam abstraídos nas interfaces de alto nível.

💡 CASOS DE USO:
- Debug de problemas de indexação
- Análise de qualidade dos dados
- Compreensão de limitações
- Otimização de performance
- Auditoria de dados
"""

import sys
from pathlib import Path

# Adicionar o diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_chroma import Chroma
from rag_demo.config import PERSIST_DIR, COLLECTION_NAME


def main():
    """Lista chunks e embeddings de forma simples e direta."""
    vect = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )

    data = vect.get(include=["documents", "metadatas", "embeddings"])
    total = len(data["ids"])
    print(f"[INFO] Total de chunks no índice: {total}\n")

    if total == 0:
        print("[INFO] Nenhum documento encontrado. Execute 'python scripts/run_ingest.py' primeiro.")
        return

    for i, (doc, meta, emb) in enumerate(
        zip(data["documents"], data["metadatas"], data["embeddings"]), 1
    ):
        print(f"--- Chunk {i} ---")
        print(f"Fonte: {meta.get('source', 'desconhecido')}")
        print(f"Texto:\n{doc}\n")
        print(f"Embedding (primeiros 10 valores): {emb[:10]}\n")


if __name__ == "__main__":
    main()
