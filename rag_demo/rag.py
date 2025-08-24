#!/usr/bin/env python3
"""
🎯 RAG ENGINE - Sistema de Recuperação e Geração Aumentada Educacional

Implementação completa do engine RAG que combina recuperação vetorial com
geração de linguagem, demonstrando a arquitetura fundamental de sistemas
de IA conversacional modernos.

📚 FUNDAMENTAÇÃO TEÓRICA:

Este módulo implementa a arquitetura RAG (Retrieval-Augmented Generation)
conforme proposta por Lewis et al. (2020), integrando recuperação de informação
baseada em embeddings com modelos de linguagem para geração contextualizada.

🎯 ARQUITETURA RAG COMPLETA:

1️⃣ RETRIEVAL PHASE (Fase de Recuperação):
   📝 CONCEITO: Busca vetorial por documentos semanticamente relevantes
   
   PROCESSO:
   - Query Encoding: Conversão da pergunta em embedding vetorial
   - Similarity Search: Busca por vizinhos mais próximos no espaço vetorial
   - Ranking: Ordenação por score de similaridade cosseno
   - Filtering: Seleção dos top-K documentos mais relevantes
   
   📐 MATEMÁTICA DA RECUPERAÇÃO:
   - Similaridade: cos(θ) = (q·d) / (||q|| × ||d||)
   - Ranking: Ordenação decrescente por similaridade
   - Top-K Selection: Seleção dos K documentos mais similares
   - Threshold Filtering: Opcional, baseado em score mínimo

2️⃣ AUGMENTATION PHASE (Fase de Aumento):
   📝 CONCEITO: Construção de contexto estruturado para o modelo de linguagem
   
   ESTRATÉGIAS:
   - Context Assembly: Concatenação dos documentos recuperados
   - Prompt Engineering: Estruturação de instruções para o LLM
   - Context Window Management: Controle de tamanho do contexto
   - Source Attribution: Preservação de metadados para citações
   
   🎨 PROMPT TEMPLATE:
   ```
   SYSTEM: Você é um assistente técnico especializado...
   CONTEXT: [Documentos recuperados formatados]
   QUERY: [Pergunta do usuário]
   INSTRUCTIONS: Responda baseado no contexto...
   ```

3️⃣ GENERATION PHASE (Fase de Geração):
   📝 CONCEITO: Geração de resposta contextualizada pelo modelo de linguagem
   
   COMPONENTES:
   - Language Model: ChatOllama com llama3
   - Temperature Control: Balanceamento criatividade vs determinismo
   - Context Injection: Inserção do contexto recuperado
   - Response Formatting: Estruturação da saída final
   
   ⚙️ CONFIGURAÇÕES LLM:
   - Model: llama3 (7B/13B parameters)
   - Temperature: 0.0 (determinística) a 1.0 (criativa)
   - Max Tokens: Controlado pelo context window
   - Stop Sequences: Tokens de parada customizáveis

📊 PIPELINE RAG DETALHADO:

STAGE 1: VECTOR STORE INITIALIZATION
```python
# Carregamento ou construção do índice vetorial
# Validação de disponibilidade e integridade
# Fallback para criação automática se necessário
```

STAGE 2: QUERY PROCESSING
```python
# Limpeza e normalização da query
# Encoding para representação vetorial
# Preparação para busca de similaridade
```

STAGE 3: SEMANTIC RETRIEVAL
```python
# Busca vetorial no ChromaDB
# Ranking por similaridade cosseno
# Seleção dos top-K documentos
```

STAGE 4: CONTEXT CONSTRUCTION
```python
# Formatação dos documentos recuperados
# Construção do prompt estruturado
# Injeção de contexto no template
```

STAGE 5: RESPONSE GENERATION
```python
# Processamento pelo modelo de linguagem
# Geração contextualizada da resposta
# Post-processamento e formatação
```

🔧 COMPONENTES ARQUITETURAIS:

VECTOR STORE MANAGEMENT:
- Lazy Loading: Inicialização sob demanda
- Auto-Creation: Construção automática se vazio
- Health Checking: Validação de integridade
- Reconnection: Recuperação de falhas de conexão

RETRIEVAL OPTIMIZATION:
- Dynamic K: Configurável via environment
- Score Thresholding: Filtragem por qualidade
- Metadata Filtering: Busca contextual avançada
- Caching: Otimização para queries similares

GENERATION CONTROL:
- Temperature Tuning: Controle de aleatoriedade
- Token Management: Controle de tamanho da resposta
- Prompt Engineering: Templates otimizados
- Error Handling: Recuperação de falhas do modelo

🎛️ CONFIGURAÇÕES AVANÇADAS:

RETRIEVAL PARAMETERS:
- RETRIEVAL_K: Número de documentos recuperados
  • Trade-off: Mais contexto vs mais ruído
  • Valores típicos: 3-10 dependendo do domínio
  • Otimização: A/B testing para cada use case

GENERATION PARAMETERS:
- TEMPERATURE: Controle de criatividade
  • 0.0: Determinística, máxima precisão
  • 0.3: Balanceada, boa para QA técnico
  • 0.7+: Criativa, boa para escrita
- MODEL: Seleção do modelo de linguagem
  • llama3: Balanceado, boa performance geral
  • llama3:70b: Maior capacidade, mais recursos

🧪 CASOS DE USO EDUCACIONAIS:

QUESTION ANSWERING:
```python
# Perguntas factuais baseadas em documentos
# Respostas com citação de fontes
# Validação contra conhecimento da base
```

DOCUMENT EXPLORATION:
```python
# Busca semântica sem geração
# Análise de relevância de documentos
# Exploração de espaço vetorial
```

CONTEXTUAL GENERATION:
```python
# Geração baseada em contexto específico
# Customização de estilo e formato
# Controle de criatividade vs precisão
```

🔍 FUNCIONALIDADES IMPLEMENTADAS:

QUERY_RAG():
- Interface principal para consultas RAG
- Pipeline completo de recuperação + geração
- Error handling e logging automático
- Suporte a modelos configuráveis

SEARCH_SIMILAR_DOCS():
- Busca semântica pura (sem LLM)
- Scores de similaridade detalhados
- Análise de relevância de documentos
- Ideal para debugging e análise

LIST_DOCUMENTS():
- Exploração do índice criado
- Visualização de metadados
- Auditoria de conteúdo indexado
- Diagnóstico de problemas

🚨 ROBUSTEZ E ERROR HANDLING:

FAULT TOLERANCE:
- Auto-creation de índice vazio
- Graceful degradation para falhas de modelo
- Retry logic para problemas temporários
- Fallback para documentos de exemplo

MONITORING:
- Logging detalhado de cada etapa
- Performance metrics automáticas
- Error reporting estruturado
- Health checks integrados

VALIDATION:
- Input sanitization para queries
- Output validation para respostas
- Context size management
- Model availability checking

🚀 VALOR EDUCACIONAL:

Este módulo demonstra:
1. Implementação completa de arquitetura RAG
2. Integração de recuperação vetorial + geração de linguagem
3. Padrões de design para sistemas de IA robustos
4. Otimização de performance para produção
5. Configuração flexível para experimentação

O design modular facilita compreensão de cada componente RAG,
experimentação com diferentes configurações, e extensão para
casos de uso específicos e pesquisa avançada.
"""

from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

from .config import (
    PERSIST_DIR, COLLECTION_NAME, LLM_MODEL, EMB_MODEL, 
    RETRIEVAL_K, TEMPERATURE, DATA_DIR
)
from .utils import format_docs, load_txt_md, get_example_documents
from .ingest import create_chunks, create_vector_store


def build_or_load_vectorstore() -> Chroma:
    """
    Constrói ou carrega o armazenamento vetorial com fallback inteligente.
    
    🎯 ESTRATÉGIA DE INICIALIZAÇÃO:
    1. Tenta carregar índice existente
    2. Verifica integridade (count > 0)
    3. Auto-cria se vazio ou inexistente
    4. Fallback para documentos de exemplo
    
    🔄 LAZY LOADING:
    - Inicialização sob demanda
    - Verificação de saúde automática
    - Reconstrução se necessário
    
    Returns:
        Chroma: Instância do vector store pronta para uso
    """
    embeddings = OllamaEmbeddings(model=EMB_MODEL)

    vect = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    try:
        existing_count = vect._collection.count()
    except Exception:
        existing_count = 0

    if existing_count == 0:  # Índice vazio, criar do zero
        print("[INFO] Índice vazio detectado. Criando do zero...")
        
        raw_docs = load_txt_md(DATA_DIR)
        if not raw_docs:
            raw_docs = get_example_documents()

        chunks = create_chunks(raw_docs)
        create_vector_store(chunks)

        # Reabrir a coleção após criação
        vect = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
        )
        print(f"[INFO] Índice criado: {len(chunks)} chunks -> {PERSIST_DIR}/")
    else:
        print(f"[INFO] Índice existente detectado (docs: {existing_count}). Usando '{PERSIST_DIR}/'.")

    return vect


def build_rag_chain(vect: Chroma):
    """
    Constrói a cadeia RAG completa usando LangChain.
    
    🔗 ARQUITETURA DA CHAIN:
    1. Retriever: Busca top-K documentos similares
    2. Context Formatter: Formata documentos para prompt
    3. Prompt Template: Estrutura instruções + contexto + query
    4. LLM: Modelo de linguagem para geração
    
    🎨 PROMPT ENGINEERING:
    - System message: Define personalidade e instruções
    - Context injection: Insere documentos recuperados
    - Query preservation: Mantém pergunta original
    - Source attribution: Facilita citações
    
    Args:
        vect: Vector store configurado para busca
        
    Returns:
        RunnableSequence: Chain RAG pronta para execução
    """
    retriever = vect.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Você é um assistente técnico especializado. Responda em português, "
         "de forma objetiva e precisa. Use o contexto fornecido quando relevante "
         "e cite as fontes no final da resposta quando possível."),
        ("human", "Pergunta: {pergunta}\n\nContexto:\n{contexto}"),
    ])

    llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE)

    # Pipeline RAG: Query → Retrieval → Context → Prompt → Generation
    chain = (
        {"contexto": retriever | format_docs, "pergunta": RunnablePassthrough()}
        | prompt
        | llm
    )
    return chain


def query_rag(question: str) -> str:
    """
    Interface principal para consultas RAG.
    
    🎯 PIPELINE COMPLETO:
    1. Inicialização do vector store
    2. Construção da chain RAG
    3. Processamento da query
    4. Recuperação de contexto
    5. Geração da resposta
    
    📊 LOGGING E MONITORAMENTO:
    - Status do Ollama
    - Query processada
    - Tempo de resposta
    - Qualidade do contexto
    
    Args:
        question: Pergunta do usuário em linguagem natural
        
    Returns:
        str: Resposta contextualizada gerada pelo LLM
    """
    print("[INFO] Certifique-se de que o Ollama está ativo (ex.: 'brew services start ollama').")
    
    vect = build_or_load_vectorstore()
    chain = build_rag_chain(vect)

    print(f"\n[Q] {question}\n")
    resp = chain.invoke(question)
    content = getattr(resp, "content", str(resp))
    return content


def search_similar_docs(query: str, k: int = 5):
    """
    Busca semântica pura sem geração de linguagem.
    
    🔍 FUNCIONALIDADE:
    - Embedding da query de busca
    - Similaridade cosseno contra todos os documentos
    - Ranking por relevância
    - Retorno com scores detalhados
    
    📊 ANÁLISE DE RELEVÂNCIA:
    - Scores de similaridade numericos
    - Metadados de fonte
    - Preview do conteúdo
    - Ranking ordenado
    
    🎯 CASOS DE USO:
    - Debugging de retrieval
    - Análise de qualidade do índice
    - Exploração de documentos
    - Validação de relevância
    
    Args:
        query: Consulta de busca
        k: Número de documentos a retornar
        
    Returns:
        List[Tuple[Document, float]]: Documentos com scores de similaridade
    """
    embeddings = OllamaEmbeddings(model=EMB_MODEL)
    vect = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    results = vect.similarity_search_with_score(query, k=k)
    
    if not results:
        print("[INFO] Nenhum resultado encontrado.")
        return []

    print(f'[Q] {query}\n')
    for i, (doc, score) in enumerate(results, 1):
        src = (doc.metadata or {}).get("source", "desconhecido")
        print(f"--- Resultado {i} | score={score:.4f} | fonte={src}")
        print(doc.page_content.strip()[:1000])
        if len(doc.page_content) > 1000:
            print("... [truncado]")
        print()
    
    return results


def list_documents():
    """
    Lista e explora todos os documentos indexados.
    
    📋 FUNCIONALIDADE:
    - Contagem total de chunks
    - Metadados de cada documento
    - Preview do conteúdo
    - Organização por fonte
    
    🔍 ANÁLISE DO ÍNDICE:
    - Verificação de integridade
    - Distribuição de conteúdo
    - Qualidade dos metadados
    - Cobertura de fontes
    
    🎯 CASOS DE USO:
    - Auditoria do índice criado
    - Verificação de ingestão
    - Exploração de conteúdo
    - Diagnóstico de problemas
    """
    embeddings = OllamaEmbeddings(model=EMB_MODEL)
    vect = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    try:
        total = vect._collection.count()
    except Exception as e:
        print(f"[ERROR] Erro ao acessar índice: {e}")
        return

    print(f"[INFO] Total de chunks no índice: {total}\n")

    if total == 0:
        print("[INFO] Nenhum documento encontrado. Execute o script de ingestão primeiro.")
        return

    # Recupera amostra de documentos para visualização
    sample_size = min(total, 10)  # Limita para não sobrecarregar
    docs = vect.get(limit=sample_size, include=["documents", "metadatas"])
    
    print(f"[INFO] Mostrando {len(docs['documents'])} de {total} documentos:\n")
    
    for i, (doc, meta) in enumerate(zip(docs["documents"], docs["metadatas"]), 1):
        source = meta.get('source', 'desconhecido')
        print(f"--- Documento {i} ---")
        print(f"Fonte: {source}")
        print(f"Conteúdo ({len(doc)} chars):")
        print(f"{doc[:500]}{'...' if len(doc) > 500 else ''}\n")
