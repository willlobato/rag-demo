#!/usr/bin/env python3
# search_docs.py
# Script para busca semântica direta (sem LLM)

import sys
from pathlib import Path

# Adicionar o diretório pai ao path para importar rag_demo
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_demo import search_similar_docs


def main():
    """Script principal para busca semântica."""
    if len(sys.argv) < 2:
        print("Uso: python search_docs.py \"sua consulta\" [k]")
        return
    
    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    try:
        search_similar_docs(query, k)
    except KeyboardInterrupt:
        print("\n[INFO] Busca interrompida pelo usuário.")
    except Exception as e:
        print(f"[ERROR] Erro durante a busca: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
