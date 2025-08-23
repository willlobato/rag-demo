#!/usr/bin/env python3
# list_docs.py
# Script para listar documentos indexados

import sys
from pathlib import Path

# Adicionar o diretório pai ao path para importar rag_demo
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_demo import list_documents


def main():
    """Script principal para listar documentos."""
    try:
        list_documents()
    except KeyboardInterrupt:
        print("\n[INFO] Listagem interrompida pelo usuário.")
    except Exception as e:
        print(f"[ERROR] Erro durante a listagem: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
