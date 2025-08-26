# 00 — Introdução: do LLM ao RAG

Bem-vindo ao trilho **LLM → RAG** deste repositório. O objetivo deste capítulo é te dar uma visão prática e direta: por que usar RAG, quais são os blocos principais, como testar localmente e o que esperar dos capítulos seguintes.

Importante: este material é um tutorial prático e hands-on com exemplos e scripts para você executar e aprender na prática.

## Por que RAG?
Modelos grandes (LLMs) são poderosos, mas:
- Podem **alucinar** (inventar respostas).
- Não têm acesso aos seus dados privados por padrão.

RAG (Retrieval-Augmented Generation) mitiga isso ao:
- **Recuperar** trechos relevantes do seu repositório de documentos.
- **Gerar** respostas condicionadas a esse contexto (reduzindo alucinações).

Benefícios: respostas mais **atuais**, **auditáveis** (com fontes) e **personalizadas**.

## O que você vai aprender neste trilho
- Fundamentos de LLM: tokens, contexto, temperatura e trade-offs.
- Embeddings e métricas de similaridade (cosseno e alternativas).
- Bases vetoriais (indexação, persistência, Chroma exemplos).
- Pipeline RAG: ingestão (chunking) → vetorização → busca → LLM.
- Guardrails e checagens (thresholds, validação, fallback).
- Boas práticas e deploy (expor via API, monitoramento).

## Como usar este material
- Leia em sequência para construir conhecimento incremental.
- Use os scripts em `scripts/` para experimentar (ingestão, consulta, métricas).
- Cada capítulo tem exemplos e referências rápidas para implementar e testar.

## Estrutura dos capítulos
- `00-introducao.md` — visão geral e checagens rápidas.
- `01-llms-basico.md` — conceitos essenciais de LLM.
- `02-embeddings-similaridade.md` — embeddings e medidas de similaridade.
- `03-bases-vetoriais.md` — como funcionam vetoriais e Chroma.
- `04-nlp-preprocessamento.md` — chunking, limpeza e normalização.
- `05-rag-basico.md` — pipeline RAG minimal.
- `06-rag-avancado.md` — estratégias avançadas e otimizações.
- `07-guardrails.md` — como evitar e manejar alucinações.
- `08-deploy.md` — opções de implantação e dicas de produção.

## Pré-requisitos (mínimos)
- Python 3.10+ e `pip`.
- Editor de código (recomendado: VSCode).
- Acesso a um runtime LLM local ou remoto (ex.: Ollama). Ollama é usado nos exemplos, mas você pode adaptar para outro provider.

Observação: o repositório foi desenvolvido em macOS, mas os conceitos são portáveis.

## Checagem rápida (smoke tests)
1) Verificar se o Ollama (ou o endpoint LLM que você usa) está ativo:

```bash
# testa o endpoint padrão do Ollama (substitua se usar outro host/port)
curl http://localhost:11434/api/tags
```

2) Executar ingestão de exemplo (vai gerar/atualizar o índice vetorial):

```bash
python scripts/run_ingest.py
```

3) Fazer uma consulta de exemplo:

```bash
python scripts/run_query.py "Como funciona RAG?"
```

Se algum comando falhar, verifique mensagens de erro e portas/variáveis de ambiente do seu provedor LLM.

## Por que usamos Python neste tutorial
- Ecossistema: Python tem a maior quantidade de bibliotecas maduras para ML/NLP (PyTorch, TensorFlow, Hugging Face, LangChain, etc.), ferramentas de vetorização e clientes para bases vetoriais (Chroma, FAISS, Milvus).
- Prototipagem rápida: sintaxe concisa e interatividade (REPL, Jupyter/Colab) aceleram experimentos e iteração de pipelines RAG.
- Integração com dados: pandas, numpy e bibliotecas de pré-processamento tornam o trabalho com texto e dados tabulares mais direto.
- Comunidade e exemplos: muitos tutoriais, notebooks e código aberto facilitam reproduzir e adaptar soluções.

Trade-offs e quando considerar Java/Go
- Java/Go oferecem vantagens em desempenho, multithreading e implantação em ambientes de alta demanda; são escolhas sólidas para serviços de produção de baixa latência.
- Recomendação prática: usar Python para pesquisa, prototipagem e pipelines RAG; empacotar ou reimplementar componentes críticos (serving, pipelines de baixa latência) em Java/Go somente quando necessário.

## Próximos passos
- Abra `01-llms-basico.md` para começar pelos fundamentos de LLMs.
- Use `scripts/run_ingest.py` para indexar os dados em `data/` e depois `scripts/run_query.py` para testar buscas.

---

Se quiser, posso também:
- ajustar os exemplos de comandos para outro provider LLM (por exemplo, OpenAI, Llama.cpp, etc.),
- ou gerar um pequeno checklist de configuração (variáveis de ambiente) adaptado ao seu ambiente local.