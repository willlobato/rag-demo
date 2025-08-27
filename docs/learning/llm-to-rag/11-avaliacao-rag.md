# Capítulo 11 — Avaliação de Sistemas RAG

Construir um sistema RAG é apenas metade da batalha. A outra metade, igualmente crucial, é saber como avaliá-lo. Como podemos ter certeza de que nosso sistema está fornecendo respostas precisas, relevantes e úteis?

Neste capítulo, vamos mergulhar nas métricas e frameworks essenciais para avaliar a qualidade de um pipeline RAG.

**Sumário:**
1. Por que a Avaliação é Crucial?
2. Métricas-Chave para Avaliação de RAG
3. Frameworks de Avaliação (RAGAs)
4. Implementando um Ciclo de Avaliação
5. Conclusão

---

## 1. Por que a Avaliação é Crucial?

Sem uma avaliação adequada, as melhorias em um sistema RAG são baseadas em "achismos". A avaliação sistemática permite:
- **Identificar Pontos Fracos:** Descobrir se o problema está na recuperação, na geração ou em ambos.
- **Otimizar Parâmetros:** Ajustar `chunk_size`, `top_k`, ou o `threshold` de similaridade com base em dados concretos.
- **Garantir a Qualidade:** Assegurar que o sistema se comporta como esperado antes de ir para produção.
- **Prevenir Regressões:** Garantir que as alterações no sistema não degradem a performance.

## 2. Métricas-Chave para Avaliação de RAG

A avaliação de RAG é multifacetada. As métricas são geralmente divididas entre a qualidade da **recuperação** e a qualidade da **geração**.

### Métricas de Recuperação

- **Context Precision:** Mede a relação sinal-ruído nos contextos recuperados. Uma pontuação alta significa que os chunks recuperados são relevantes para a pergunta.
- **Context Recall:** Mede se todos os chunks relevantes da base de conhecimento foram recuperados. Uma pontuação alta significa que o sistema não deixou passar informações importantes.

### Métricas de Geração

- **Faithfulness (Fidelidade):** Mede o quão factualmente correta é a resposta gerada em relação ao contexto fornecido. Uma pontuação baixa indica que o LLM está "alucinando" ou inventando informações não presentes nos documentos.
- **Answer Relevancy (Relevância da Resposta):** Mede o quão relevante é a resposta gerada para a pergunta do usuário. Uma resposta pode ser factualmente correta, mas não responder diretamente à pergunta.
- **Answer Correctness (Correção da Resposta):** Mede a precisão da resposta em comparação com uma resposta de referência (ground truth). Isso é útil quando se tem um conjunto de dados de perguntas e respostas validadas.

## 3. Frameworks de Avaliação (RAGAs)

Felizmente, não precisamos implementar essas métricas do zero. Ferramentas como o **RAGAs** (RAG Assessment) simplificam enormemente o processo.

**RAGAs** é uma biblioteca Python que oferece um framework para avaliar pipelines RAG usando as métricas que discutimos. Ele funciona gerando um conjunto de dados sintético ou usando um conjunto de dados existente e, em seguida, calculando as pontuações de `faithfulness`, `context precision`, etc.

**Exemplo de como o RAGAs funciona:**
1.  **Preparação do Dataset:** Você fornece um conjunto de documentos. O RAGAs pode ajudar a gerar um conjunto de perguntas e respostas de referência a partir desses documentos.
2.  **Execução do Pipeline:** Você executa seu sistema RAG com as perguntas do dataset para obter as respostas e os contextos recuperados.
3.  **Cálculo das Métricas:** O RAGAs avalia as saídas do seu pipeline e calcula as pontuações para cada métrica.

```python
# Exemplo simplificado de uso do RAGAs
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

# Supondo que você tenha um dataset com perguntas, respostas, contextos e ground truths
dataset = Dataset.from_dict({
    "question": ["What is RAG?"],
    "answer": ["RAG is a technique..."],
    "contexts": [["RAG stands for Retrieval-Augmented Generation..."]],
    "ground_truth": ["Retrieval-Augmented Generation (RAG) is a technique..."]
})

result = evaluate(
    dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)

# O resultado será um dicionário com as pontuações
print(result)
```

## 4. Implementando um Ciclo de Avaliação

A avaliação não deve ser um evento único, mas um processo contínuo.

**Ciclo de Avaliação Iterativo:**
1.  **Coletar Dados:** Reúna um conjunto de perguntas e respostas de referência.
2.  **Executar Avaliação:** Use o RAGAs para obter um baseline da performance do seu sistema.
3.  **Analisar Resultados:** Identifique as métricas com pontuação mais baixa.
    -   `Context Precision` baixa? Tente otimizar o chunking ou o modelo de embedding.
    -   `Faithfulness` baixa? Tente ajustar o prompt do LLM para ser mais restritivo ao contexto.
    -   `Answer Relevancy` baixa? O LLM pode estar se desviando do assunto. Ajuste o prompt.
4.  **Implementar Melhorias:** Faça uma alteração de cada vez.
5.  **Reavaliar:** Execute a avaliação novamente e compare os resultados para ver se a alteração teve um impacto positivo.

## 5. Conclusão

A avaliação é uma parte indispensável do desenvolvimento de sistemas RAG robustos e confiáveis. Métricas como `faithfulness`, `answer relevancy`, `context precision` e `context recall` fornecem uma visão clara dos pontos fortes e fracos do seu pipeline. Ferramentas como o RAGAs tornam a implementação desse processo muito mais acessível, permitindo que você itere e melhore seu sistema com base em evidências concretas.
