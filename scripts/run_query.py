#!/usr/bin/env python3
# run_query.py
# Script para fazer consultas RAG

import sys
from pathlib import Path

# Adicionar o diretório pai ao path para importar rag_demo
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_demo import query_rag


def main():
    """Script principal para consultas RAG."""
    question = "O que é RAG e como ele funciona?"
    
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
    
    try:
        response = query_rag(question)
        print(response)
    except KeyboardInterrupt:
        print("\n[INFO] Consulta interrompida pelo usuário.")
    except Exception as e:
        print(f"[ERROR] Erro durante a consulta: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
