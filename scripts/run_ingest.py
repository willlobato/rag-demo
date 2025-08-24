#!/usr/bin/env python3
# run_ingest.py
#!/usr/bin/env python3
"""
📚 RUN INGEST - PIPELINE DE INGESTÃO EDUCACIONAL PARA DOCUMENTOS RAG

Este script implementa o processo fundamental de ingestão de documentos em um sistema RAG,
demonstrando cada etapa do pipeline desde o carregamento até a indexação vetorial.

🎯 PROCESSO DE INGESTÃO EXPLICADO:

1️⃣ DOCUMENT LOADING (Carregamento):
   - Leitura de arquivos de diferentes formatos (TXT, MD, PDF)
   - Validação de encoding e integridade
   - Tratamento de erros e arquivos corrompidos

2️⃣ TEXT PREPROCESSING (Pré-processamento):
   - Limpeza de caracteres especiais e formatação
   - Normalização de espaços e quebras de linha
   - Remoção de metadados desnecessários

3️⃣ DOCUMENT SPLITTING (Chunking):
   - Divisão em chunks de tamanho otimizado
   - Preservação de contexto semântico
   - Overlap estratégico entre chunks

4️⃣ EMBEDDING GENERATION (Vetorização):
   - Conversão de texto em representações vetoriais
   - Uso de modelos pré-treinados (nomic-embed-text)
   - Normalização dos vetores para busca eficiente

5️⃣ VECTOR INDEXING (Indexação):
   - Armazenamento no vector store (ChromaDB)
   - Criação de índices para busca rápida
   - Associação de metadados aos chunks

📐 CONCEITOS MATEMÁTICOS:

CHUNKING STRATEGY:
- Tamanho ótimo: Balance entre contexto e precisão
- Overlap: Preserva informação que cruza fronteiras
- Fórmula: chunk_size = optimal_context_window / embedding_efficiency

EMBEDDING SPACE:
- Dimensionalidade: Tipicamente 768-1536 dimensões
- Normalização: ||v|| = 1 para comparações justas
- Distribuição: Embeddings bem distribuídos no espaço n-dimensional

INDEXING EFFICIENCY:
- Complexidade: O(n log n) para construção do índice
- Busca: O(log n) para retrieval em índices otimizados
- Memória: Linear com número de documentos

🔧 CONFIGURAÇÕES IMPORTANTES:

CHUNK_SIZE (padrão: 500 caracteres):
- Pequeno (200-400): Alta precisão, contexto limitado
- Médio (500-800): Balanceado para uso geral
- Grande (1000+): Muito contexto, possível ruído

CHUNK_OVERLAP (padrão: 50 caracteres):
- Preserva informação que cruza fronteiras de chunks
- Evita perda de conceitos importantes
- Trade-off: redundância vs completude

EMBEDDING_MODEL:
- nomic-embed-text: 768 dimensões, boa qualidade geral
- Alternativas: bge-large, e5-large, sentence-transformers
- Considerações: Velocidade vs qualidade vs tamanho

🚀 USO EDUCACIONAL:
Este script demonstra os fundamentos da preparação de dados para RAG,
incluindo as decisões de design e trade-offs envolvidos em cada etapa.
"""

import sys
from pathlib import Path

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
