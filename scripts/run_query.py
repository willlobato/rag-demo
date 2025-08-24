#!/usr/bin/env python3
"""
🎯 RAG QUERY RUNNER - Interface de Consulta Educacional

Sistema completo de execução de consultas RAG com explicações pedagógicas
detalhadas sobre cada componente do pipeline de processamento.

📚 FUNDAMENTOS EDUCACIONAIS:

Este script demonstra a implementação prática de um sistema RAG completo,
servindo como laboratório educacional para compreender como consultas são
processadas e respostas são geradas usando recuperação de informação e
geração aumentada.

🎯 ARQUITETURA DO PIPELINE RAG:

1️⃣ QUERY PROCESSING (Processamento da Consulta):
   📝 CONCEITO: Primeira etapa de normalização e preparação da entrada
   
   COMPONENTES:
   - Limpeza textual: Remoção de caracteres especiais e normalização
   - Detecção de intenção: Classificação do tipo de pergunta
   - Expansão semântica: Adição de termos relacionados (opcional)
   - Validação de entrada: Verificação de formato e conteúdo
   
   🔬 TÉCNICAS APLICADAS:
   - Text preprocessing: Tokenização e normalização Unicode
   - Intent classification: Padrões regex ou ML para categorização
   - Query expansion: Sinônimos e termos relacionados via thesaurus
   - Input validation: Sanitização e verificação de segurança

2️⃣ QUERY EMBEDDING (Vetorização da Consulta):
   📝 CONCEITO: Conversão da pergunta em representação vetorial densa
   
   PROCESSO:
   - Tokenização: Divisão em unidades semânticas
   - Encoding: Transformação em IDs de vocabulário
   - Embedding: Mapeamento para espaço vetorial denso
   - Normalização: Ajuste para compatibilidade com índice
   
   📐 MATEMÁTICA:
   - Modelo usado: sentence-transformers/all-MiniLM-L6-v2
   - Dimensões: 384 (vetor denso)
   - Normalização L2: ||v|| = 1 para compatibilidade cosseno
   - Representação: v ∈ ℝ³⁸⁴, onde cada dimensão captura aspectos semânticos

3️⃣ SIMILARITY SEARCH (Busca por Similaridade):
   📝 CONCEITO: Recuperação de documentos relevantes via proximidade vetorial
   
   ALGORITMO:
   - Indexação: Estrutura de dados otimizada (HNSW no ChromaDB)
   - Busca: Algoritmo de vizinhos mais próximos (k-NN)
   - Ranking: Ordenação por score de similaridade
   - Filtragem: Aplicação de thresholds opcionais
   
   📐 MATEMÁTICA DA SIMILARIDADE:
   - Métrica: Similaridade cosseno cos(θ) = (q·d) / (||q|| × ||d||)
   - Range: [0, 1] onde 1 = idêntico, 0 = ortogonal
   - Interpretação: Ângulo entre vetores no espaço semântico
   - Complexidade: O(log n) com índice HNSW vs O(n) busca linear

4️⃣ CONTEXT ASSEMBLY (Montagem do Contexto):
   📝 CONCEITO: Combinação inteligente dos documentos recuperados
   
   ESTRATÉGIAS:
   - Concatenação simples: União sequencial dos chunks
   - Ordenação híbrida: Relevância + cronologia + autoridade
   - Deduplicação: Remoção de informações redundantes
   - Truncagem inteligente: Preservação dos chunks mais importantes
   
   🎚️ PARÂMETROS:
   - Max tokens: Limite do contexto (normalmente 75% do window)
   - Ranking weights: Pesos para diferentes critérios de ordenação
   - Overlap handling: Tratamento de informações sobrepostas
   - Quality thresholds: Filtros de qualidade do conteúdo

5️⃣ PROMPT ENGINEERING (Construção do Prompt):
   � CONCEITO: Design de instruções estruturadas para o LLM
   
   COMPONENTES:
   - System prompt: Instruções gerais e personalidade do assistente
   - Context injection: Inserção dos documentos recuperados
   - Task specification: Definição clara da tarefa esperada
   - Output formatting: Especificação do formato de resposta
   
   🎨 TEMPLATE STRUCTURE:
   ```
   SYSTEM: Você é um assistente especializado...
   CONTEXT: [Documentos recuperados]
   QUERY: [Pergunta do usuário]
   INSTRUCTIONS: Responda baseado apenas no contexto...
   ```

6️⃣ RESPONSE GENERATION (Geração da Resposta):
   📝 CONCEITO: Processamento pelo modelo de linguagem e pós-processamento
   
   ETAPAS:
   - LLM processing: Geração via Ollama llama3
   - Response validation: Verificação de qualidade e relevância
   - Post-processing: Formatação e limpeza da saída
   - Confidence scoring: Avaliação da confiança na resposta
   
   ⚙️ CONFIGURAÇÕES:
   - Temperature: Controle de criatividade (0.0-1.0)
   - Max tokens: Limite de tamanho da resposta
   - Stop sequences: Tokens de parada para controle de geração
   - Repetition penalty: Prevenção de repetições excessivas

📊 MÉTRICAS E AVALIAÇÃO:

RETRIEVAL METRICS:
- Precision@K: Proporção de documentos relevantes nos top-K
- Recall@K: Cobertura dos documentos relevantes
- MRR (Mean Reciprocal Rank): Qualidade do ranking
- NDCG: Normalized Discounted Cumulative Gain

GENERATION METRICS:
- BLEU Score: Similaridade com respostas de referência
- ROUGE Score: Overlap de n-gramas
- BERTScore: Similaridade semântica via embeddings
- Human evaluation: Avaliação manual de qualidade

🔧 CONFIGURAÇÕES AVANÇADAS:

RETRIEVAL PARAMETERS:
- k: Número de documentos recuperados (padrão: 5)
  • Trade-off: Mais contexto vs mais ruído
  • Recomendação: 3-10 dependendo da complexidade do domínio
  
- similarity_threshold: Filtro de relevância (padrão: 0.0)
  • Uso: Eliminar documentos pouco relevantes
  • Valores típicos: 0.3-0.7 dependendo da precisão necessária
  
- max_context_length: Limite de tokens do contexto
  • Prevenção: Overflow no modelo de linguagem
  • Estratégia: Truncagem dos documentos menos relevantes

GENERATION PARAMETERS:
- temperature: Controle de aleatoriedade
  • 0.0: Determinística, mais precisa
  • 1.0: Criativa, mais variada
  
- top_p: Nucleus sampling para controle de qualidade
- presence_penalty: Incentivo à diversidade vocabular
- frequency_penalty: Redução de repetições

🎭 TIPOS DE CONSULTAS E ESTRATÉGIAS:

FACTUAL QUERIES (Consultas Factuais):
- Exemplo: "Qual é a capital do Brasil?"
- Estratégia: Busca por chunks com informações específicas
- Expectativa: Resposta direta e precisa
- Avaliação: Exatidão factual

PROCEDURAL QUERIES (Consultas Procedimentais):
- Exemplo: "Como fazer login no sistema?"
- Estratégia: Recuperar documentação de processos
- Expectativa: Sequência clara de passos
- Avaliação: Completude e clareza das instruções

COMPARATIVE QUERIES (Consultas Comparativas):
- Exemplo: "Qual a diferença entre X e Y?"
- Estratégia: Recuperar documentos sobre ambos os temas
- Expectativa: Análise balanceada e estruturada
- Avaliação: Imparcialidade e abrangência

EXPLANATORY QUERIES (Consultas Explicativas):
- Exemplo: "Por que isso acontece?"
- Estratégia: Buscar contexto conceitual amplo
- Expectativa: Explicação detalhada com fundamentos
- Avaliação: Profundidade e clareza conceitual

EXPLORATORY QUERIES (Consultas Exploratórias):
- Exemplo: "Me fale sobre inteligência artificial"
- Estratégia: Recuperação diversificada de aspectos
- Expectativa: Visão abrangente do tópico
- Avaliação: Cobertura e organização das informações

🧪 LABORATÓRIO EDUCACIONAL:

EXPERIMENTOS SUGERIDOS:
1. Variação do parâmetro K: Observe como diferentes números de documentos
   afetam a qualidade e relevância das respostas
   
2. Ajuste de similarity_threshold: Teste diferentes limiares para
   entender o trade-off entre precisão e recall
   
3. Comparação de embeddings: Experimente diferentes modelos de embedding
   para ver como afetam a recuperação
   
4. Análise de prompt templates: Modifique os templates para entender
   como o formato afeta as respostas

DEBUGGING E MONITORAMENTO:
- Log de documentos recuperados para análise de relevância
- Medição de tempos de resposta em cada etapa
- Análise de distribuição de scores de similaridade
- Tracking de tipos de consulta mais comuns

🚀 OBJETIVOS PEDAGÓGICOS:

Este script serve como:
1. Demonstração prática de um pipeline RAG completo
2. Laboratório para experimentação com parâmetros
3. Base para compreensão de trade-offs em sistemas de recuperação
4. Exemplo de boas práticas em implementação de chatbots
5. Ferramenta para análise de qualidade de respostas

O código é projetado para ser educativo, mostrando cada decisão de design
e permitindo fácil experimentação com diferentes configurações.
"""

import sys
from pathlib import Path

# Adicionar o diretório pai ao path para importar rag_demo
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_demo import query_rag


def main():
    """Script principal para consultas RAG."""
    question = "O que é RAG e como ele funciona?"
    
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
    
    try:
        response = query_rag(question)
        print(response)
    except KeyboardInterrupt:
        print("\n[INFO] Consulta interrompida pelo usuário.")
    except Exception as e:
        print(f"[ERROR] Erro durante a consulta: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
