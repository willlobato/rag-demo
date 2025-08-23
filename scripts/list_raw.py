#!/usr/bin/env python3
# list_raw.py
# Lista chunks salvos e seus embeddings (parciais) no Chroma.

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
