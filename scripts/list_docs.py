#!/usr/bin/env python3
"""
📋 LIST DOCS - EXPLORADOR EDUCACIONAL DE DOCUMENTOS INDEXADOS

Este script fornece uma interface educacional para explorar e compreender
os documentos armazenados no vector store, demonstrando como os dados
são organizados e indexados em um sistema RAG.

🎯 FUNCIONALIDADES EDUCACIONAIS:

📚 EXPLORAÇÃO DE METADADOS:
- Lista todos os documentos com suas propriedades
- Mostra source, chunk_id, e outras informações estruturais
- Demonstra como os metadados são preservados durante indexação
- Explica a importância dos metadados para filtragem e organização

📊 ESTATÍSTICAS DE COLEÇÃO:
- Número total de chunks por documento
- Distribuição de tamanhos e tipos
- Análise de fontes e categorias
- Métricas de cobertura da base de conhecimento

🚀 USO EDUCACIONAL:
Este script ajuda a compreender como os dados são organizados
internamente em um sistema RAG, demonstrando as melhores práticas
de estruturação e indexação de documentos.
"""

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
