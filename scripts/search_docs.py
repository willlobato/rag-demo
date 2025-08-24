#!/usr/bin/env python3
"""
🔎 SEARCH DOCS - DEMONSTRAÇÃO EDUCACIONAL DE BUSCA VETORIAL

Este script demonstra como funciona a busca vetorial em sistemas RAG,
permitindo explorar diretamente o mecanismo de recuperação de documentos
sem a camada de geração de respostas.

🎯 FUNCIONALIDADES EDUCACIONAIS:

🔍 BUSCA VETORIAL PURA:
- Demonstra busca por similaridade semântica
- Mostra scores de similaridade em tempo real
- Permite experimentar com diferentes queries
- Explica como embeddings determinam relevância

📊 ANÁLISE DE RESULTADOS:
- Ranking dos documentos por similaridade
- Scores numéricos para interpretação
- Comparação entre diferentes tipos de query
- Identificação de padrões de recuperação

🚀 USO EDUCACIONAL:
Este script é ideal para compreender os fundamentos da busca
vetorial, permitindo experimentação direta com o core engine
do sistema RAG antes de adicionar a camada de geração.
"""

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
