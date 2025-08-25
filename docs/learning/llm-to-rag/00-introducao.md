# 00 — Introdução: do LLM ao RAG

Bem-vindo ao trilho **LLM → RAG** deste repositório. A ideia é te levar do zero (fundamentos de LLMs e embeddings) até um **RAG sólido** com **Ollama + Chroma + LangChain**, documentando tudo para você e para sua equipe.

## Por que RAG?
Modelos grandes “sabem muito”, mas podem **alucinar** e não conhecem seus **dados privados**. O RAG resolve isso:
1. **Recupera** trechos relevantes dos seus documentos.
2. **Gera** a resposta usando *apenas* esse contexto.

Resultado: respostas **atuais**, **auditáveis** (com fontes) e **customizadas** aos seus dados.

## O que você vai aprender
- Conceitos de **LLM** (tokens, contexto, temperatura, limites).
- **Embeddings** e **similaridade** (coseno e alternativas).
- **Bases vetoriais** (como indexar e recuperar).
- **Pipeline RAG**: ingestão → busca → LLM → guardrails.
- **Boas práticas** (chunking, thresholds, métricas).
- **Deploy** (expondo via API).

## Como usar este “livro”
- Cada capítulo é curto e prático.
- Sempre que possível, há **scripts** prontos neste repo para testar.
- Leia na ordem, mas sinta-se livre para pular para os tópicos de interesse.

## Pré-requisitos (mínimos)
- Python 3.10+, `pip`, e um editor (VSCode).
- macOS com **Ollama** (ou outra máquina compatível).
- Noções de linha de comando.

### Checagem rápida
```bash
# Verificar Ollama rodando
curl http://localhost:11434/api/tags

# Testar ingestão e consulta no projeto
python scripts/run_ingest.py
python scripts/run_query.py "Como funciona RAG?"