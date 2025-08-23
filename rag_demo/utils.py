# utils.py
# Funções utilitárias para carregamento de documentos e embeddings

from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def load_txt_md(data_dir: Path) -> List[Document]:
    """Carrega arquivos TXT e MD do diretório especificado."""
    docs = []
    for p in data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
                docs.append(Document(page_content=text, metadata={"source": str(p)}))
            except Exception as e:
                print(f"[WARN] Falha ao ler arquivo {p}: {e}")
    return docs


def load_pdfs(data_dir: Path) -> List[Document]:
    """Carrega arquivos PDF do diretório especificado."""
    docs = []
    for p in data_dir.rglob("*.pdf"):
        try:
            for d in PyPDFLoader(str(p)).load():
                d.metadata = {"source": str(p)}
                docs.append(d)
        except Exception as e:
            print(f"[WARN] Falha ao ler PDF {p}: {e}")
    return docs


def get_example_documents() -> List[Document]:
    """Retorna documentos de exemplo quando não há arquivos na pasta data/."""
    return [
        Document(
            page_content=(
                "RAG combina busca vetorial e LLM. Divida os documentos em chunks, "
                "crie embeddings e recupere para construir a resposta."
            ),
            metadata={"source": "guia.md"},
        ),
        Document(
            page_content=(
                "Exemplo: otimização de login com paralelismo e cache (Infinispan)."
            ),
            metadata={"source": "login.md"},
        ),
    ]


def format_docs(docs: List[Document]) -> str:
    """Formata documentos para uso no prompt RAG."""
    if not docs:
        return "(sem contexto relevante)"
    return "\n---\n".join(
        f"{d.page_content}\n(fonte: {d.metadata.get('source')})" for d in docs
    )
