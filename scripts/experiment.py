#!/usr/bin/env python3
# experiment.py
# Script para experimentação com diferentes configurações RAG

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
