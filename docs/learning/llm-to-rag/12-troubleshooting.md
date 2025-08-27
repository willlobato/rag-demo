# Capítulo 12 — Guia de Troubleshooting para Sistemas RAG

Mesmo com a melhor arquitetura, sistemas RAG podem falhar. As respostas podem ser irrelevantes, incorretas ou o sistema pode simplesmente não encontrar a informação. Este capítulo é um guia prático para diagnosticar e resolver os problemas mais comuns em pipelines RAG.

**Objetivo:** Fornecer um checklist de problemas e soluções para ajudar a depurar e otimizar seu sistema RAG.

**Sumário:**
1. Problema: A busca não retorna chunks relevantes
2. Problema: A resposta do LLM é genérica ou "não sei"
3. Problema: O LLM está alucinando (inventando informações)
4. Problema: A resposta é correta, mas não responde à pergunta
5. Problema: O processo de ingestão é muito lento
6. Checklist Rápido de Debugging

---

### 1. Problema: A busca não retorna chunks relevantes

Você faz uma pergunta, mas os documentos recuperados da base vetorial parecem não ter relação com o que foi perguntado.

- **Causas Prováveis:**
    - **Chunking Ineficaz:** Os chunks são muito grandes e genéricos, ou muito pequenos e sem contexto.
    - **Modelo de Embedding Inadequado:** O modelo de embedding que você está usando pode não ser bom em capturar a semântica do seu domínio específico.
    - **Pergunta Mal Formulada:** A pergunta usa uma terminologia muito diferente da que existe nos seus documentos.

- **Soluções:**
    - **Ajuste o Chunking:** Experimente diferentes estratégias (`RecursiveCharacterTextSplitter` vs. `SemanticChunking`), tamanhos (`chunk_size`) e sobreposições (`chunk_overlap`).
    - **Teste Outros Embeddings:** Modelos diferentes têm performances diferentes. Teste modelos como `mxbai-embed-large` ou outros específicos para o seu idioma.
    - **Use Re-ranking:** Adicione uma camada de re-ranking (como `Cohere Rerank` ou `bge-reranker`) para reordenar os `top_k` chunks recuperados e trazer os mais relevantes para o topo.
    - **Query Expansion:** Use um LLM para reescrever a pergunta do usuário de várias formas diferentes e busque por todas as variações.

### 2. Problema: A resposta do LLM é genérica ou "não sei"

Os chunks recuperados parecem corretos, mas o LLM se recusa a responder ou dá uma resposta evasiva.

- **Causas Prováveis:**
    - **Contexto Insuficiente:** O `top_k` (número de chunks recuperados) é muito baixo e não fornece informação suficiente.
    - **Threshold de Similaridade Muito Alto:** Seu sistema está filtrando chunks que, embora não perfeitamente alinhados, poderiam conter a resposta.
    - **Prompt Restritivo Demais:** O prompt instrui o LLM a não responder se não tiver 100% de certeza.

- **Soluções:**
    - **Aumente o `top_k`:** Recupere mais documentos (ex: de 3 para 5) para dar mais "matéria-prima" ao LLM.
    - **Ajuste o Threshold:** Use o `threshold_optimizer.py` para encontrar um valor ideal que não filtre resultados úteis.
    - **Refine o Prompt:** Permita que o LLM sintetize informações de múltiplos chunks. Ex: "Use os documentos abaixo para responder à pergunta. Se a resposta estiver espalhada por vários trechos, sintetize-a."

### 3. Problema: O LLM está alucinando (inventando informações)

A resposta parece plausível, mas ao verificar as fontes, você descobre que a informação foi inventada ou distorcida.

- **Causas Prováveis:**
    - **Prompt Muito Aberto:** O prompt não instrui o LLM a se ater estritamente ao contexto.
    - **Temperatura Alta:** A configuração de `temperature` do LLM está alta, incentivando a criatividade em vez da precisão.
    - **Contexto Ambíguo:** Os chunks recuperados são contraditórios ou pouco claros.

- **Soluções:**
    - **Prompt de "Grounding":** Seja explícito no prompt. Ex: "**Responda APENAS com base no contexto fornecido. Se a resposta não estiver no contexto, diga que você não sabe.**"
    - **Reduza a `temperature`:** Para tarefas baseadas em fatos, use uma `temperature` de 0.0 ou 0.1.
    - **Melhore a Qualidade dos Dados:** Limpe e normalize seus documentos de origem para evitar ambiguidades.

### 4. Problema: A resposta é correta, mas não responde à pergunta

O LLM extrai um fato correto do contexto, mas esse fato não é a resposta para a pergunta do usuário. (Ex: Pergunta: "Por que o céu é azul?", Resposta: "O céu é a atmosfera acima da Terra.").

- **Causas Prováveis:**
    - **Problema de `Answer Relevancy`:** O LLM se perdeu no contexto e focou na informação errada.
    - **Prompt Pouco Específico:** O prompt não guia o LLM sobre como usar o contexto para construir a resposta.

- **Soluções:**
    - **Engenharia de Prompt:** Adicione uma instrução final no prompt: "Após gerar a resposta, verifique se ela responde diretamente à pergunta original do usuário."
    - **Re-ranking:** Um bom re-ranker pode ajudar a priorizar chunks que são mais diretamente relevantes para a pergunta.

### 5. Problema: O processo de ingestão é muito lento

Leva horas para indexar seus documentos.

- **Causas Prováveis:**
    - **Processamento em Série:** Você está processando um documento de cada vez.
    - **Modelo de Embedding Pesado em CPU:** Modelos de embedding grandes podem ser lentos sem uma GPU.
    - **Escrita em Batch Pequeno:** Você está escrevendo na base vetorial chunk por chunk.

- **Soluções:**
    - **Paralelização:** Use `multiprocessing` para processar múltiplos documentos em paralelo.
    - **Use GPU:** Se disponível, certifique-se de que sua biblioteca de embedding está configurada para usar a GPU.
    - **Batch Writing:** Agrupe os chunks e escreva-os na base vetorial em lotes maiores (ex: 100 de cada vez).

---

## Checklist Rápido de Debugging

Quando algo der errado, verifique nesta ordem:
1.  **A Pergunta:** Ela está clara? Tente reformulá-la.
2.  **Os Chunks Recuperados:** Imprima os documentos que a busca vetorial retornou. Eles são relevantes? Se não, o problema está na **recuperação**.
3.  **O Prompt Final:** Imprima o prompt completo que é enviado ao LLM (com o contexto inserido). Ele está formatado corretamente?
4.  **A Resposta do LLM:** Se os chunks e o prompt estão bons, mas a resposta é ruim, o problema está na **geração**.
5.  **Os Dados de Origem:** A informação que você procura realmente existe na sua base de conhecimento?
