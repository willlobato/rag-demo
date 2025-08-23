# rag.py
# Sistema RAG principal com Ollama + ChromaDB

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
    """Constrói ou carrega o armazenamento vetorial."""
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

    if existing_count == 0:  # índice vazio, criar do zero
        print("[INFO] Índice vazio detectado. Criando do zero...")
        
        raw_docs = load_txt_md(DATA_DIR)
        if not raw_docs:
            raw_docs = get_example_documents()

        chunks = create_chunks(raw_docs)
        create_vector_store(chunks)

        # Reabrir a coleção
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
    """Constrói a cadeia RAG completa."""
    retriever = vect.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Você é um assistente técnico. Responda em português, de forma objetiva. "
         "Use o contexto quando útil e cite as fontes no final."),
        ("human", "Pergunta: {pergunta}\n\nContexto:\n{contexto}"),
    ])

    llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE)

    chain = (
        {"contexto": retriever | format_docs, "pergunta": RunnablePassthrough()}
        | prompt
        | llm
    )
    return chain


def query_rag(question: str) -> str:
    """Faz uma consulta RAG e retorna a resposta."""
    print("[INFO] Certifique-se de que o Ollama está ativo (ex.: 'brew services start ollama').")
    
    vect = build_or_load_vectorstore()
    chain = build_rag_chain(vect)

    print(f"\n[Q] {question}\n")
    resp = chain.invoke(question)
    content = getattr(resp, "content", str(resp))
    return content


def search_similar_docs(query: str, k: int = 5):
    """Busca documentos similares sem usar LLM."""
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
        print()
    
    return results


def list_documents():
    """Lista todos os documentos indexados."""
    vect = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )

    total = vect._collection.count()
    print(f"[INFO] Total de chunks no índice: {total}\n")

    if total == 0:
        print("[INFO] Nenhum documento encontrado. Execute o script de ingestão primeiro.")
        return

    docs = vect.get(include=["documents", "metadatas"])
    for i, (doc, meta) in enumerate(zip(docs["documents"], docs["metadatas"]), 1):
        print(f"--- Documento {i} ---")
        print(f"Fonte: {meta.get('source')}")
        print(f"Conteúdo:\n{doc}\n")
