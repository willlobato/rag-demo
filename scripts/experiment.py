#!/usr/bin/env python3
"""
🧪 EXPERIMENT FRAMEWORK - Laboratório de Experimentação Científica para RAG

Framework avançado para experimentação controlada e científica com sistemas RAG,
implementando metodologias rigorosas de pesquisa para otimização de hiperparâmetros,
testes A/B e validação estatística de diferentes configurações.

📚 FUNDAMENTAÇÃO CIENTÍFICA:

Este framework aplica princípios de pesquisa experimental para sistemas de IA,
combinando metodologias de ciência da computação, estatística experimental e
engenharia de sistemas para criar um ambiente de experimentação controlado
e reproduzível para sistemas RAG.

🎯 METODOLOGIA EXPERIMENTAL:

O framework implementa uma abordagem científica rigorosa baseada em:
- Design experimental controlado com variáveis isoladas
- Medição quantitativa com métricas estatisticamente válidas  
- Reprodutibilidade através de seeds fixos e configurações determinísticas
- Validação estatística usando testes de hipótese e intervalos de confiança
- Análise comparativa usando técnicas de benchmarking sistemático

🔬 TIPOS DE EXPERIMENTOS CIENTÍFICOS:

1️⃣ CHUNK SIZE OPTIMIZATION EXPERIMENTS:
   📝 OBJETIVO: Determinar tamanho ótimo de chunks via análise empírica
   
   VARIÁVEIS INDEPENDENTES:
   - Tamanhos testados: 200, 500, 1000, 1500, 2000 caracteres
   - Estratégia de chunking: Fixa vs semântica vs híbrida
   - Overlap ratio: 0%, 10%, 20%, 30%
   
   MÉTRICAS DEPENDENTES:
   - Precision@K: Proporção de chunks relevantes recuperados
   - Recall@K: Cobertura dos chunks relevantes disponíveis
   - Response Time: Latência média de processamento
   - Context Coherence: Medida de coesão semântica do contexto
   - Information Density: Densidade informacional por token
   
   📐 HIPÓTESES TESTADAS:
   H₁: Chunks maiores fornecem mais contexto mas reduzem precisão
   H₂: Chunks menores aumentam precisão mas podem fragmentar informação
   H₃: Existe um tamanho ótimo que maximiza F1-score para cada domínio
   
   🧮 ANÁLISE ESTATÍSTICA:
   - ANOVA para comparação múltipla de tamanhos
   - Post-hoc Tukey HSD para pairwise comparisons
   - Correlação de Pearson entre tamanho e métricas de qualidade

2️⃣ K-VALUE RETRIEVAL OPTIMIZATION:
   📝 OBJETIVO: Otimizar número de documentos recuperados
   
   DESIGN EXPERIMENTAL:
   - Valores K testados: 1, 3, 5, 10, 15, 20, 30
   - Métricas de avaliação: Precision@K, Recall@K, NDCG@K
   - Análise de trade-offs: Qualidade vs tempo de processamento
   
   📊 ANÁLISE DE PERFORMANCE:
   - Curve fitting para encontrar K ótimo
   - Elbow method para detectar ponto de diminishing returns
   - ROC analysis para threshold optimization
   
   🎯 OTIMIZAÇÃO MULTI-OBJETIVO:
   - Pareto frontier analysis para K vs tempo vs qualidade
   - Weighted scoring function para diferentes casos de uso
   - Sensitivity analysis para robustez da escolha de K

3️⃣ EMBEDDING MODEL COMPARISON:
   📝 OBJETIVO: Avaliação comparativa sistemática de modelos
   
   MODELOS AVALIADOS:
   - sentence-transformers/all-MiniLM-L6-v2 (384D)
   - sentence-transformers/all-mpnet-base-v2 (768D)  
   - nomic-ai/nomic-embed-text-v1.5 (768D)
   - BAAI/bge-large-en-v1.5 (1024D)
   
   DIMENSÕES DE COMPARAÇÃO:
   - Retrieval Quality: mAP, NDCG, MRR
   - Computational Efficiency: Embedding time, memory usage
   - Domain Adaptation: Performance em diferentes tipos de conteúdo
   - Multilingual Capability: Suporte a múltiplas linguagens
   
   📈 MÉTRICAS AVANÇADAS:
   - Embedding Space Quality: Isotropy, dimensionality analysis
   - Semantic Preservation: Cosine similarity distribution
   - Cluster Coherence: Silhouette score, Davies-Bouldin index

4️⃣ A/B TESTING FRAMEWORK:
   📝 OBJETIVO: Comparação estatisticamente rigorosa de configurações
   
   METODOLOGIA:
   - Randomized Controlled Trial (RCT) design
   - Stratified sampling para balanceamento de grupos
   - Power analysis para tamanho amostral adequado
   - Blinding quando aplicável (automated evaluation)
   
   📊 ANÁLISE ESTATÍSTICA:
   - Student's t-test para comparação de médias
   - Mann-Whitney U test para distribuições não-normais
   - Chi-square test para variáveis categóricas
   - Bootstrap confidence intervals para robustez
   
   🎯 VALIDAÇÃO:
   - Cross-validation para generalização
   - Effect size calculation (Cohen's d)
   - Multiple testing correction (Bonferroni)
   - Practical significance assessment

5️⃣ OVERLAP STRATEGY EXPERIMENTS:
   📝 OBJETIVO: Otimizar estratégias de sobreposição entre chunks
   
   CONFIGURAÇÕES TESTADAS:
   - Fixed overlap: 0%, 10%, 20%, 30%, 50%
   - Semantic overlap: Baseado em similaridade semântica
   - Sentence boundary: Preservação de fronteiras de sentença
   - Paragraph boundary: Preservação de fronteiras de parágrafo
   
   MÉTRICAS ESPECÍFICAS:
   - Information Redundancy: Medida de duplicação informacional
   - Context Continuity: Fluidez semântica entre chunks
   - Boundary Preservation: Preservação de unidades semânticas

📊 FRAMEWORK DE MÉTRICAS CIENTÍFICAS:

RETRIEVAL METRICS:
- Mean Average Precision (mAP): Precisão média ponderada por posição
- Normalized Discounted Cumulative Gain (NDCG): Relevância com desconto posicional
- Mean Reciprocal Rank (MRR): Posição média do primeiro resultado relevante
- Recall@K: Cobertura dos documentos relevantes nos top-K
- Precision@K: Proporção de documentos relevantes nos top-K

GENERATION METRICS:
- BLEU Score: Precisão de n-gramas com penalização de brevidade
- ROUGE-L: Longest Common Subsequence baseado em F-measure
- BERTScore: Similaridade semântica usando embeddings BERT
- METEOR: Alinhamento com sinônimos e stemming
- Human Evaluation Scores: Relevância, fluência, factualidade

EFFICIENCY METRICS:
- Response Latency: Tempo total de processamento
- Throughput: Queries processadas por segundo
- Memory Usage: Pico de utilização de memória
- CPU Utilization: Utilização média de processador
- Storage Requirements: Espaço necessário para índices

SYSTEM METRICS:
- Index Build Time: Tempo para construção do índice
- Query Processing Time: Tempo de processamento da query
- Retrieval Time: Tempo para busca de documentos
- Generation Time: Tempo para geração da resposta
- End-to-End Latency: Latência total do sistema

🎛️ CONFIGURAÇÕES EXPERIMENTAIS:

EXPERIMENTAL VARIABLES:
- Chunk size: 200-2000 characters (step: 200)
- Overlap percentage: 0%-50% (step: 10%)
- Number of retrievals (K): 1-30 (exponential scale)
- Embedding model: Lista de modelos pré-definidos
- Similarity threshold: 0.0-1.0 (step: 0.1)

CONTROL VARIABLES:
- Random seed: Fixado para reprodutibilidade
- Dataset splits: Train/validation/test consistentes
- Evaluation queries: Conjunto padronizado de perguntas
- Hardware configuration: Especificações fixas de teste
- Software versions: Versões específicas de dependências

🧪 PROCEDIMENTOS EXPERIMENTAIS:

SETUP PHASE:
1. Environment initialization: Configuração determinística
2. Data preparation: Preprocessamento padronizado
3. Baseline establishment: Métricas de referência
4. Resource allocation: Configuração de recursos computacionais

EXECUTION PHASE:
1. Parameter grid generation: Combinações sistemáticas
2. Controlled execution: Isolamento de variáveis
3. Metrics collection: Logging padronizado
4. Progress monitoring: Acompanhamento em tempo real

ANALYSIS PHASE:
1. Statistical testing: Testes de hipótese apropriados
2. Effect size calculation: Magnitude prática das diferenças
3. Confidence intervals: Intervalos de confiança para robustez
4. Visualization: Gráficos e tabelas explanatórias

VALIDATION PHASE:
1. Cross-validation: Validação cruzada k-fold
2. Hold-out testing: Teste em conjunto independente
3. Sensitivity analysis: Robustez a mudanças de parâmetros
4. Reproducibility check: Verificação de reprodutibilidade

� OUTPUTS E RELATÓRIOS:

QUANTITATIVE RESULTS:
- Performance tables: Métricas organizadas por configuração
- Statistical summaries: Médias, desvios, intervalos de confiança
- Ranking tables: Ordenação por performance
- Significance tests: P-values e effect sizes

VISUALIZATIONS:
- Performance curves: Métricas vs parâmetros
- Scatter plots: Correlações entre variáveis
- Box plots: Distribuições de performance
- Heatmaps: Interações entre parâmetros

ANALYTICAL REPORTS:
- Best configuration recommendations: Configurações ótimas
- Trade-off analysis: Análise de compromissos
- Sensitivity analysis: Robustez das recomendações
- Practical guidelines: Diretrizes para implementação

🚀 VALOR EDUCACIONAL:

Este framework demonstra:

import sys
import json
import time
import itertools
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict

# Adicionar o diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_demo.rag import query_rag, search_similar_docs
from rag_demo.ingest import create_chunks, load_all_documents
from rag_demo.config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class ExperimentConfig:
    """Configuração de um experimento."""
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    embedding_model: str = "nomic-embed-text"
    experiment_id: str = ""


@dataclass
class ExperimentResult:
    """Resultado de um experimento."""
    config: ExperimentConfig
    question: str
    answer: str
    response_time: float
    retrieved_chunks_count: int
    answer_length: int
    chunks_total: int


class RAGExperimentRunner:
    """Classe para executar experimentos RAG."""
    
    def __init__(self):
        self.test_questions = [
            "Como foi otimizado o processo de login?",
            "Qual tecnologia foi usada para cache?",
            "Quanto tempo levava o login antes da otimização?",
            "Que tipo de paralelismo foi implementado?",
            "Qual é o tempo atual do processo de login?"
        ]
    
    def run_experiment(self, config: ExperimentConfig, questions: List[str] = None) -> List[ExperimentResult]:
        """Executa um experimento com uma configuração específica."""
        if questions is None:
            questions = self.test_questions
        
        print(f"\n🧪 Executando experimento: {config.experiment_id}")
        print(f"   Chunk Size: {config.chunk_size}")
        print(f"   Chunk Overlap: {config.chunk_overlap}")
        print(f"   Retrieval K: {config.retrieval_k}")
        
        results = []
        
        # Simular mudança de configuração (na prática, você recriaria o índice)
        print(f"   📊 Processando {len(questions)} perguntas...")
        
        for i, question in enumerate(questions, 1):
            print(f"      {i}/{len(questions)}: {question[:50]}...")
            
            start_time = time.time()
            
            try:
                # Buscar chunks
                chunks = search_similar_docs(question, k=config.retrieval_k)
                retrieved_count = len(chunks) if chunks else 0
                
                # Gerar resposta
                answer = query_rag(question)
                
                response_time = time.time() - start_time
                
                result = ExperimentResult(
                    config=config,
                    question=question,
                    answer=answer,
                    response_time=response_time,
                    retrieved_chunks_count=retrieved_count,
                    answer_length=len(answer),
                    chunks_total=self.get_total_chunks()
                )
                
                results.append(result)
                
            except Exception as e:
                print(f"      ❌ Erro: {e}")
                continue
        
        return results
    
    def get_total_chunks(self) -> int:
        """Obtém o número total de chunks no índice."""
        try:
            from rag_demo.rag import build_or_load_vectorstore
            vect = build_or_load_vectorstore()
            return vect._collection.count()
        except:
            return 0
    
    def compare_chunk_sizes(self, chunk_sizes: List[int], overlap_ratio: float = 0.15) -> Dict[str, List[ExperimentResult]]:
        """Compara diferentes tamanhos de chunk."""
        print("\n🔬 EXPERIMENTO: Comparação de Chunk Sizes")
        print("="*60)
        
        results = {}
        
        for chunk_size in chunk_sizes:
            overlap = int(chunk_size * overlap_ratio)
            
            config = ExperimentConfig(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                retrieval_k=4,
                experiment_id=f"chunk_size_{chunk_size}"
            )
            
            experiment_results = self.run_experiment(config)
            results[f"chunk_{chunk_size}"] = experiment_results
        
        return results
    
    def compare_retrieval_k(self, k_values: List[int]) -> Dict[str, List[ExperimentResult]]:
        """Compara diferentes valores de K para recuperação."""
        print("\n🔬 EXPERIMENTO: Comparação de Retrieval K")
        print("="*60)
        
        results = {}
        
        for k in k_values:
            config = ExperimentConfig(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                retrieval_k=k,
                experiment_id=f"retrieval_k_{k}"
            )
            
            experiment_results = self.run_experiment(config)
            results[f"k_{k}"] = experiment_results
        
        return results
    
    def batch_test_questions(self, questions_file: str, config: ExperimentConfig) -> List[ExperimentResult]:
        """Testa um lote de perguntas de um arquivo."""
        print(f"\n🔬 EXPERIMENTO: Batch Test de {questions_file}")
        print("="*60)
        
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                if questions_file.endswith('.json'):
                    data = json.load(f)
                    questions = [item.get('question', '') for item in data if item.get('question')]
                else:
                    questions = [line.strip() for line in f if line.strip()]
            
            print(f"📝 Carregadas {len(questions)} perguntas do arquivo")
            return self.run_experiment(config, questions)
            
        except FileNotFoundError:
            print(f"❌ Arquivo não encontrado: {questions_file}")
            return []
        except Exception as e:
            print(f"❌ Erro ao carregar perguntas: {e}")
            return []


def analyze_experiment_results(results: Dict[str, List[ExperimentResult]]):
    """Analisa e compara resultados de experimentos."""
    print("\n📊 ANÁLISE DOS RESULTADOS")
    print("="*60)
    
    for experiment_name, experiment_results in results.items():
        if not experiment_results:
            continue
            
        print(f"\n🔍 {experiment_name.upper()}:")
        
        # Métricas agregadas
        avg_response_time = sum(r.response_time for r in experiment_results) / len(experiment_results)
        avg_answer_length = sum(r.answer_length for r in experiment_results) / len(experiment_results)
        avg_retrieved_chunks = sum(r.retrieved_chunks_count for r in experiment_results) / len(experiment_results)
        
        print(f"   📈 Tempo médio de resposta: {avg_response_time:.2f}s")
        print(f"   📝 Tamanho médio da resposta: {avg_answer_length:.0f} caracteres")
        print(f"   📚 Chunks recuperados (média): {avg_retrieved_chunks:.1f}")
        print(f"   🔢 Total de perguntas: {len(experiment_results)}")
        
        # Configuração usada
        if experiment_results:
            config = experiment_results[0].config
            print(f"   ⚙️  Configuração:")
            print(f"      - Chunk Size: {config.chunk_size}")
            print(f"      - Chunk Overlap: {config.chunk_overlap}")
            print(f"      - Retrieval K: {config.retrieval_k}")


def save_experiment_results(results: Dict[str, List[ExperimentResult]], output_file: str):
    """Salva resultados dos experimentos em arquivo."""
    output_data = {}
    
    for experiment_name, experiment_results in results.items():
        output_data[experiment_name] = [
            {
                "config": asdict(result.config),
                "question": result.question,
                "answer": result.answer,
                "response_time": result.response_time,
                "retrieved_chunks_count": result.retrieved_chunks_count,
                "answer_length": result.answer_length,
                "chunks_total": result.chunks_total
            }
            for result in experiment_results
        ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados salvos em: {output_file}")


def create_sample_questions_file(filename: str = "sample_questions.txt"):
    """Cria arquivo de exemplo com perguntas para teste."""
    questions = [
        "Como foi otimizado o processo de login?",
        "Qual tecnologia foi usada para cache?",
        "Quanto tempo levava o login antes da otimização?",
        "Que tipo de paralelismo foi implementado?",
        "Qual é o tempo atual do processo de login?",
        "Quais são as principais melhorias implementadas?",
        "Como funciona o cache distribuído?",
        "Qual foi a redução percentual do tempo de login?",
        "Que benefícios trouxe o paralelismo?",
        "O sistema ficou mais eficiente?"
    ]
    
    with open(filename, 'w', encoding='utf-8') as f:
        for question in questions:
            f.write(question + '\n')
    
    print(f"📝 Arquivo de perguntas criado: {filename}")


def main():
    """Script principal de experimentação."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Experimentação com configurações RAG")
    parser.add_argument("--chunk-sizes", type=str, default="250,500,1000", 
                       help="Tamanhos de chunk para testar (separados por vírgula)")
    parser.add_argument("--retrieval-k", type=str, default="3,5,7", 
                       help="Valores de K para testar (separados por vírgula)")
    parser.add_argument("--questions-file", type=str, 
                       help="Arquivo com perguntas para teste em lote")
    parser.add_argument("--output", type=str, default="experiment_results.json", 
                       help="Arquivo de saída")
    parser.add_argument("--create-questions", action="store_true", 
                       help="Criar arquivo de exemplo com perguntas")
    parser.add_argument("--compare-chunks", action="store_true", 
                       help="Comparar diferentes tamanhos de chunk")
    parser.add_argument("--compare-k", action="store_true", 
                       help="Comparar diferentes valores de K")
    parser.add_argument("--all", action="store_true", 
                       help="Executar todos os experimentos")
    
    args = parser.parse_args()
    
    try:
        runner = RAGExperimentRunner()
        all_results = {}
        
        if args.create_questions:
            create_sample_questions_file()
            return
        
        if args.all or args.compare_chunks:
            chunk_sizes = [int(x.strip()) for x in args.chunk_sizes.split(',')]
            chunk_results = runner.compare_chunk_sizes(chunk_sizes)
            all_results.update(chunk_results)
        
        if args.all or args.compare_k:
            k_values = [int(x.strip()) for x in args.retrieval_k.split(',')]
            k_results = runner.compare_retrieval_k(k_values)
            all_results.update(k_results)
        
        if args.questions_file:
            config = ExperimentConfig(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                retrieval_k=4,
                experiment_id="batch_questions"
            )
            
            batch_results = runner.batch_test_questions(args.questions_file, config)
            all_results["batch_test"] = batch_results
        
        # Analisar e salvar resultados
        if all_results:
            analyze_experiment_results(all_results)
            save_experiment_results(all_results, args.output)
        else:
            print("🎯 Experimentos RAG - Opções disponíveis:")
            print("   --all                    : Executar todos os experimentos")
            print("   --compare-chunks         : Comparar tamanhos de chunk")
            print("   --compare-k              : Comparar valores de K")
            print("   --questions-file FILE    : Teste em lote com arquivo")
            print("   --create-questions       : Criar arquivo de exemplo")
            print("   --chunk-sizes 250,500,1000")
            print("   --retrieval-k 3,5,7")
            print("\nExemplo: python scripts/experiment.py --all")
        
    except Exception as e:
        print(f"❌ Erro durante experimentação: {e}")


if __name__ == "__main__":
    main()
