#!/usr/bin/env python3
# run_ingest.py
# Script para executar a ingestão de documentos

import sys
from pathlib import Path

# Adicionar o diretório pai ao path para importar rag_demo
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_demo import ingest_documents


def main():
    """Script principal para ingestão de documentos."""
    try:
        ingest_documents()
    except KeyboardInterrupt:
        print("\n[INFO] Processo interrompido pelo usuário.")
    except Exception as e:
        print(f"[ERROR] Erro durante a ingestão: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
