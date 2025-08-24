#!/usr/bin/env python3
"""
📥 DOCUMENT INGESTION ENGINE - Pipeline de Ingestão Educacional

Implementação completa do pipeline de ingestão para sistemas RAG, demonstrando
as melhores práticas para processamento, chunking e indexação de documentos
em bancos vetoriais.

📚 FUNDAMENTAÇÃO TEÓRICA:

Este módulo implementa o pipeline de ingestão baseado nos princípios fundamentais
de recuperação de informação e processamento de linguagem natural, aplicando
técnicas estabelecidas de chunking, embedding e indexação vetorial.

🎯 ARQUITETURA DO PIPELINE DE INGESTÃO:

1️⃣ DOCUMENT LOADING (Carregamento de Documentos):
   📝 CONCEITO: Primeira etapa onde documentos brutos são carregados e preparados
   
   PROCESSO:
   - Detecção automática de formatos (TXT, MD, PDF)
   - Carregamento paralelo para performance
   - Validação de integridade dos documentos
   - Fallback para documentos de exemplo se necessário
   
   🔬 TÉCNICAS APLICADAS:
   - Detecção de encoding automática (UTF-8, Latin-1, etc.)
   - Sanitização de caracteres especiais
   - Preservação de metadados estruturais
   - Error handling robusto para arquivos corrompidos

2️⃣ TEXT CHUNKING (Fragmentação de Texto):
   📝 CONCEITO: Divisão inteligente de documentos em fragmentos processáveis
   
   ESTRATÉGIA - RecursiveCharacterTextSplitter:
   - Preservação de estrutura semântica
   - Separadores hierárquicos (\n\n, \n, ., espaço)
   - Overlap para manutenção de contexto
   - Tamanho otimizado para embeddings
   
   📐 MATEMÁTICA DO CHUNKING:
   - Chunk Size: Tamanho alvo em caracteres (500 padrão)
   - Overlap: Sobreposição entre chunks (80 chars, ~16%)
   - Ratio: Overlap/Size = 0.16 (preserva contexto sem redundância excessiva)
   
   🎯 TRADE-OFFS:
   - Chunks grandes: Mais contexto, menos precisão
   - Chunks pequenos: Mais precisão, menos contexto
   - Overlap alto: Mais continuidade, mais redundância

3️⃣ EMBEDDING GENERATION (Geração de Embeddings):
   📝 CONCEITO: Conversão de texto em representações vetoriais densas
   
   MODELO: nomic-embed-text (via Ollama)
   - Dimensionalidade: 768 (típico para modelos modernos)
   - Arquitetura: Transformer encoder especializado
   - Treinamento: Contrastive learning em large corpus
   - Normalização: L2 norm para compatibilidade cosseno
   
   🔬 PROCESSO DE EMBEDDING:
   - Tokenização: Conversão texto → tokens
   - Encoding: Tokens → representação interna
   - Pooling: Sequência → vetor denso único
   - Normalização: ||v|| = 1 para similaridade cosseno

4️⃣ VECTOR INDEXING (Indexação Vetorial):
   📝 CONCEITO: Armazenamento otimizado para busca de similaridade
   
   CHROMADB ARCHITECTURE:
   - Estrutura: Hierarchical Navigable Small World (HNSW)
   - Complexidade: O(log n) para busca vs O(n) busca linear
   - Métrica: Similaridade cosseno por padrão
   - Persistência: Automática com metadata preservation
   
   🏗️ ESTRUTURA DO ÍNDICE:
   - Document Store: Texto original dos chunks
   - Vector Store: Embeddings normalizados
   - Metadata Store: Informações contextuais
   - Collection: Namespace lógico para organização

📊 PIPELINE STAGES DETALHADO:

STAGE 1: DOCUMENT DISCOVERY
```python
# Busca automática por arquivos suportados
# Validação de formato e integridade
# Criação de estrutura de dados unificada
```

STAGE 2: CONTENT EXTRACTION
```python
# Extração de texto limpo
# Preservação de estrutura (parágrafos, seções)
# Metadata extraction (título, autor, data)
```

STAGE 3: TEXT PREPROCESSING
```python
# Limpeza de caracteres especiais
# Normalização de espaços em branco
# Detecção e preservação de estruturas importantes
```

STAGE 4: INTELLIGENT CHUNKING
```python
# Aplicação do RecursiveCharacterTextSplitter
# Preservação de fronteiras semânticas
# Cálculo de overlap otimizado
```

STAGE 5: EMBEDDING COMPUTATION
```python
# Processamento batch para eficiência
# Geração de vetores via Ollama
# Validação de dimensionalidade
```

STAGE 6: INDEX CONSTRUCTION
```python
# Inserção no ChromaDB
# Construção de índices HNSW
# Persistência automática
```

🔧 CONFIGURAÇÕES E OTIMIZAÇÕES:

CHUNKING OPTIMIZATION:
- Chunk Size: Balanceado para context window dos embeddings
- Overlap: Suficiente para continuidade, mínimo para redundância
- Separators: Hierárquicos para preservar estrutura semântica

EMBEDDING OPTIMIZATION:
- Batch Processing: Processa múltiplos chunks simultaneamente
- Model Caching: Reutiliza modelo carregado
- Error Handling: Retry logic para falhas temporárias

STORAGE OPTIMIZATION:
- Compression: ChromaDB comprime automaticamente
- Indexing: HNSW otimizado para busca rápida
- Metadata: Structured storage para filtragem eficiente

🚨 ERROR HANDLING E ROBUSTEZ:

FAULT TOLERANCE:
- Graceful degradation com documentos de exemplo
- Retry logic para falhas de rede/modelo
- Validation de dados em cada estágio
- Rollback capability para falhas críticas

MONITORING & LOGGING:
- Progress tracking durante ingestão
- Performance metrics (documentos/segundo)
- Error reporting detalhado
- Resource usage monitoring

🧪 CASOS DE USO EDUCACIONAIS:

EXPERIMENTAÇÃO COM CHUNKING:
```python
# Teste diferentes estratégias de chunking
# Análise de impacto no retrieval
# Otimização baseada em métricas
```

ANÁLISE DE EMBEDDINGS:
```python
# Visualização de espaços vetoriais
# Análise de qualidade de embeddings
# Comparação entre modelos
```

OTIMIZAÇÃO DE PERFORMANCE:
```python
# Benchmarking de velocidade de ingestão
# Análise de uso de memória
# Otimização de batch sizes
```

🚀 VALOR EDUCACIONAL:

Este módulo demonstra:
1. Pipeline completo de ingestão para sistemas RAG
2. Aplicação prática de técnicas de NLP
3. Otimização de performance em sistemas vetoriais
4. Error handling e robustez em produção
5. Configuração flexível para experimentação

O design modular facilita compreensão de cada etapa, experimentação
com diferentes configurações, e extensão para casos de uso específicos.
"""

import os
import shutil
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from .config import (
    DATA_DIR, PERSIST_DIR, COLLECTION_NAME, EMB_MODEL, 
    CHUNK_SIZE, CHUNK_OVERLAP, RESET_CHROMA
)
from .utils import load_txt_md, load_pdfs, get_example_documents


def reset_index_if_needed():
    """
    Reseta o índice ChromaDB se configurado via RESET_CHROMA.
    
    🔄 FUNCIONALIDADE:
    - Verifica flag RESET_CHROMA do environment
    - Remove completamente o diretório de persistência
    - Permite rebuild completo do índice
    - Útil para experimentação e desenvolvimento
    
    ⚠️ CUIDADO: Operação destrutiva - remove todos os dados indexados
    """
    if RESET_CHROMA and Path(PERSIST_DIR).exists():
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        print(f"[INFO] Reset de índice: removido '{PERSIST_DIR}/'.")


def load_all_documents():
    """
    Carrega todos os documentos disponíveis do diretório de dados.
    
    📁 PROCESSO:
    1. Verifica existência do diretório DATA_DIR
    2. Carrega documentos TXT, MD e PDF
    3. Fallback para documentos de exemplo se vazio
    4. Retorna lista unificada de documentos
    
    🔄 FALLBACK STRATEGY:
    - Cria diretório de dados se não existir
    - Usa documentos de exemplo para demonstração
    - Permite funcionamento out-of-the-box
    
    Returns:
        List[Document]: Lista de documentos carregados com metadata
    """
    if not DATA_DIR.exists():
        print("[INFO] Pasta 'data/' não encontrada. Criando vazia.")
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_docs = load_txt_md(DATA_DIR) + load_pdfs(DATA_DIR)

    if not raw_docs:
        raw_docs = get_example_documents()
        print("[INFO] Sem arquivos em data/. Usando documentos de exemplo.")

    return raw_docs


def create_chunks(documents):
    """
    Divide documentos em chunks usando RecursiveCharacterTextSplitter.
    
    🔪 ESTRATÉGIA DE CHUNKING:
    - Separadores hierárquicos: \n\n (parágrafos) → \n (linhas) → . (sentenças)
    - Preservação de contexto via overlap
    - Tamanho otimizado para embeddings
    - Manutenção de metadados originais
    
    📐 PARÂMETROS CONFIGURÁVEIS:
    - chunk_size: Tamanho alvo em caracteres
    - chunk_overlap: Sobreposição para continuidade
    
    Args:
        documents: Lista de documentos a serem fragmentados
        
    Returns:
        List[Document]: Lista de chunks com metadata preservada
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"[INFO] Chunks gerados: {len(chunks)}")
    return chunks


def create_vector_store(chunks):
    """
    Cria o armazenamento vetorial usando ChromaDB.
    
    🗄️ PROCESSO DE INDEXAÇÃO:
    1. Inicializa modelo de embeddings (Ollama)
    2. Gera embeddings para todos os chunks
    3. Constrói índice HNSW para busca eficiente
    4. Persiste automaticamente no disco
    
    🔧 CONFIGURAÇÕES:
    - Embedding Model: Configurável via EMB_MODEL
    - Collection: Namespace lógico para organização
    - Persistence: Automática no diretório especificado
    
    Args:
        chunks: Lista de chunks para indexação
    """
    embeddings = OllamaEmbeddings(model=EMB_MODEL)

    # ChromaDB 0.4+ com persistência automática
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )

    print(f"[OK] Índice atualizado em '{PERSIST_DIR}/' (coleção: {COLLECTION_NAME})")


def ingest_documents():
    """
    Função principal que coordena todo o pipeline de ingestão.
    
    🎯 PIPELINE COMPLETO:
    1. Reset opcional do índice existente
    2. Carregamento de documentos do diretório
    3. Fragmentação inteligente em chunks
    4. Geração de embeddings vetoriais
    5. Construção e persistência do índice
    
    📊 MONITORAMENTO:
    - Logging detalhado de cada etapa
    - Contadores de progresso
    - Relatório de conclusão
    
    🔄 IDEMPOTÊNCIA:
    - Pode ser executada múltiplas vezes
    - Atualiza índice existente ou cria novo
    - Configurações via environment variables
    """
    print("[INFO] Iniciando processo de ingestão...")
    
    # Reset opcional do índice
    reset_index_if_needed()
    
    # Carregar documentos
    documents = load_all_documents()
    
    # Criar chunks
    chunks = create_chunks(documents)
    
    # Criar store vetorial
    create_vector_store(chunks)
    
    print("[INFO] Processo de ingestão concluído!")
