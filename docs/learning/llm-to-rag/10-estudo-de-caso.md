# Capítulo 10 — Estudo de Caso: Criando um Chatbot de Perguntas e Respostas sobre o Projeto

Neste capítulo, vamos aplicar tudo o que aprendemos em um estudo de caso prático: a criação de um chatbot capaz de responder perguntas sobre a documentação do nosso próprio projeto RAG.

**Objetivo:** Construir e avaliar um sistema RAG que utiliza os arquivos `.md` deste repositório como base de conhecimento.

**Sumário:**
1. Definição do Problema e Escopo
2. Preparação dos Dados (Nossa Documentação)
3. Executando o Pipeline de Ingestão
4. Realizando Consultas e Analisando Respostas
5. Limitações e Próximos Passos
6. Conclusão do Estudo de Caso

---

## 1. Definição do Problema e Escopo

O desafio é simples: como podemos facilitar o acesso à informação contida nos diversos arquivos de documentação deste projeto? Em vez de ler todos os arquivos `.md`, um usuário poderia simplesmente "perguntar" ao sistema.

**Escopo:**
- **Base de Conhecimento:** Todos os arquivos de texto (`.txt`, `.md`) no diretório `data/` e `docs/`.
- **Perguntas-alvo:** "Como funciona o RAG?", "Para que serve o script `evaluate_rag.py`?", "Qual a diferença entre `chunk_size` e `chunk_overlap`?".
- **Sistema:** Utilizaremos os scripts `run_ingest.py` e `run_query.py` que desenvolvemos.

## 2. Preparação dos Dados

Para este estudo de caso, nossa base de conhecimento já está pronta: é a própria documentação do projeto. O primeiro passo seria garantir que os documentos mais relevantes estejam em um local que nosso script de ingestão possa ler.

Por padrão, nosso `ingest.py` lê do diretório `data/`. Vamos garantir que uma cópia dos documentos de `docs/` esteja lá ou ajustar o script para ler de múltiplos diretórios. Para simplificar, vamos assumir que o `sistema_completo.txt` em `data/` contém uma versão consolidada da documentação.

## 3. Executando o Pipeline de Ingestão

Com os dados no lugar, o próximo passo é indexá-los em nossa base vetorial. Isso é feito executando o script de ingestão.

```bash
# Este comando irá ler os documentos, dividi-los em chunks,
# gerar os embeddings e armazená-los no ChromaDB.
python scripts/run_ingest.py
```

Ao final da execução, você terá uma base vetorial (`db/`) pronta para receber consultas.

## 4. Realizando Consultas e Analisando Respostas

Agora, a parte mais interessante: testar o sistema com perguntas reais.

**Exemplo de Consulta 1: Pergunta Geral**

```bash
python scripts/run_query.py "O que é RAG e por que é útil?"
```

- **Resposta Esperada:** Uma definição clara de RAG, mencionando a recuperação de documentos e a geração aumentada, baseada no conteúdo de `00-introducao.md` e `05-rag-basico.md`.
- **Análise:** A resposta foi precisa? Ela citou as fontes corretas (metadados dos chunks)? Houve alguma alucinação?

**Exemplo de Consulta 2: Pergunta Específica sobre Código**

```bash
python scripts/run_query.py "Qual a função do script 'threshold_optimizer.py'?"
```

- **Resposta Esperada:** Uma explicação de que o script serve para encontrar o limiar (threshold) de similaridade ideal para filtrar resultados irrelevantes, com base na documentação de referência.
- **Análise:** O sistema conseguiu encontrar a informação em um arquivo de documentação de scripts? A resposta foi técnica e correta?

**Exemplo de Consulta 3: Pergunta com Termos Ambíguos**

```bash
python scripts/run_query.py "Como funcionam os guardrails?"
```

- **Resposta Esperada:** Uma explicação sobre as checagens de segurança, validação de entrada/saída e o uso de thresholds, conforme descrito em `07-guardrails.md`.
- **Análise:** O sistema conseguiu diferenciar "guardrails" no contexto de RAG de outros significados? A resposta foi focada no que a documentação apresenta?

## 5. Limitações e Próximos Passos

Durante os testes, podemos encontrar algumas limitações:
- **Respostas Genéricas:** Se os chunks recuperados não forem específicos o suficiente, o LLM pode gerar respostas vagas.
- **Informação Fragmentada:** A resposta pode ser construída a partir de múltiplos chunks que, juntos, não formam uma explicação coerente.
- **Falha na Recuperação:** Para perguntas muito específicas ou com jargões não presentes na documentação, o sistema pode não encontrar chunks relevantes.

**Próximos Passos para Melhoria:**
- **Otimização de Chunking:** Experimentar diferentes tamanhos e sobreposições de chunks.
- **Re-ranking:** Implementar um re-ranker para reordenar os chunks recuperados antes de enviá-los ao LLM.
- **Engenharia de Prompt:** Melhorar o prompt para instruir o LLM a lidar melhor com a informação recebida.

## 6. Conclusão do Estudo de Caso

Este estudo de caso demonstrou como podemos usar a arquitetura RAG para criar uma ferramenta poderosa de busca e resposta sobre uma base de conhecimento customizada. Mesmo com uma implementação básica, conseguimos obter respostas relevantes para perguntas complexas sobre nosso projeto.

As limitações encontradas são naturais e servem como um guia para as otimizações que exploramos nos capítulos de RAG Avançado e Guardrails. O mais importante é que agora temos um sistema funcional que pode ser iterativamente melhorado.
