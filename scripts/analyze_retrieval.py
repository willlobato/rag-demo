#!/usr/bin/env python3
# analyze_retrieval.py
# Script para análise detalhada da qualidade de recuperação

import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
from collections import Counter, defaultdict

# Adicionar o diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_demo.rag import build_or_load_vectorstore, search_similar_docs


class RetrievalAnalyzer:
    """Analisador de qualidade de recuperação."""
    
    def __init__(self):
        self.vect = build_or_load_vectorstore()
        self.total_chunks = self.vect._collection.count()
        print(f"📚 ChromaDB carregado com {self.total_chunks} chunks")
    
    def analyze_query_retrieval(self, query: str, k_values: List[int] = None) -> Dict[str, Any]:
        """Analisa recuperação para uma query específica."""
        if k_values is None:
            k_values = [1, 3, 5, 10]
        
        print(f"\n🔍 Analisando recuperação para: '{query}'")
        
        analysis = {
            'query': query,
            'k_analysis': {},
            'chunk_details': []
        }
        
        # Testar diferentes valores de K
        for k in k_values:
            if k > self.total_chunks:
                continue
                
            results = search_similar_docs(query, k=k)
            
            if results:
                scores = [score for doc, score in results]
                chunks = [doc.page_content for doc, score in results]
                sources = [doc.metadata.get('source', 'unknown') for doc, score in results]
                
                analysis['k_analysis'][k] = {
                    'retrieved_count': len(results),
                    'avg_score': np.mean(scores),
                    'min_score': min(scores),
                    'max_score': max(scores),
                    'score_std': np.std(scores),
                    'sources': list(set(sources)),
                    'source_distribution': dict(Counter(sources))
                }
                
                # Detalhes dos chunks apenas para k=5
                if k == 5:
                    for i, (doc, score) in enumerate(results):
                        analysis['chunk_details'].append({
                            'rank': i + 1,
                            'score': score,
                            'source': doc.metadata.get('source', 'unknown'),
                            'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                        })
        
        return analysis
    
    def calculate_recall_at_k(self, queries_with_relevant: List[Dict[str, Any]]) -> Dict[int, float]:
        """Calcula Recall@K para um conjunto de queries com documentos relevantes conhecidos."""
        print("\n📊 Calculando Recall@K...")
        
        k_values = [1, 3, 5, 10]
        recall_at_k = {k: [] for k in k_values}
        
        for query_data in queries_with_relevant:
            query = query_data['query']
            relevant_chunks = set(query_data['relevant_chunks'])  # IDs dos chunks relevantes
            
            print(f"   🎯 {query[:50]}...")
            
            for k in k_values:
                if k > self.total_chunks:
                    continue
                
                results = search_similar_docs(query, k=k)
                retrieved_chunks = set(range(len(results)))  # Simplificado - na prática seria ID real
                
                if relevant_chunks:
                    recall = len(retrieved_chunks.intersection(relevant_chunks)) / len(relevant_chunks)
                    recall_at_k[k].append(recall)
        
        # Calcular médias
        avg_recall_at_k = {}
        for k, recalls in recall_at_k.items():
            if recalls:
                avg_recall_at_k[k] = np.mean(recalls)
        
        return avg_recall_at_k
    
    def analyze_chunk_popularity(self, queries: List[str], k: int = 5) -> Dict[str, Any]:
        """Analisa quais chunks são mais frequentemente recuperados."""
        print(f"\n🏆 Analisando popularidade dos chunks (k={k})...")
        
        chunk_retrieval_count = defaultdict(int)
        source_retrieval_count = defaultdict(int)
        total_retrievals = 0
        
        for query in queries:
            print(f"   🔍 {query[:50]}...")
            results = search_similar_docs(query, k=k)
            
            for doc, score in results:
                # Usar conteúdo como ID (simplificado)
                chunk_id = doc.page_content[:100]  # Primeiros 100 chars como ID
                chunk_retrieval_count[chunk_id] += 1
                
                source = doc.metadata.get('source', 'unknown')
                source_retrieval_count[source] += 1
                
                total_retrievals += 1
        
        # Chunks mais populares
        popular_chunks = sorted(chunk_retrieval_count.items(), key=lambda x: x[1], reverse=True)
        popular_sources = sorted(source_retrieval_count.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_queries': len(queries),
            'total_retrievals': total_retrievals,
            'unique_chunks_retrieved': len(chunk_retrieval_count),
            'popular_chunks': popular_chunks[:10],  # Top 10
            'popular_sources': popular_sources,
            'chunk_coverage': len(chunk_retrieval_count) / self.total_chunks if self.total_chunks > 0 else 0
        }
    
    def analyze_score_distribution(self, queries: List[str], k: int = 5) -> Dict[str, Any]:
        """Analisa distribuição dos scores de similaridade."""
        print(f"\n📈 Analisando distribuição de scores (k={k})...")
        
        all_scores = []
        score_by_rank = defaultdict(list)
        
        for query in queries:
            results = search_similar_docs(query, k=k)
            
            for rank, (doc, score) in enumerate(results):
                all_scores.append(score)
                score_by_rank[rank + 1].append(score)
        
        if not all_scores:
            return {'error': 'Nenhum score encontrado'}
        
        # Estatísticas gerais
        analysis = {
            'total_scores': len(all_scores),
            'mean_score': np.mean(all_scores),
            'median_score': np.median(all_scores),
            'std_score': np.std(all_scores),
            'min_score': min(all_scores),
            'max_score': max(all_scores),
            'score_ranges': {
                'excellent': sum(1 for s in all_scores if s <= 0.3),  # Baixa distância = alta similaridade
                'good': sum(1 for s in all_scores if 0.3 < s <= 0.5),
                'fair': sum(1 for s in all_scores if 0.5 < s <= 0.7),
                'poor': sum(1 for s in all_scores if s > 0.7)
            }
        }
        
        # Estatísticas por ranking
        rank_stats = {}
        for rank, scores in score_by_rank.items():
            if scores:
                rank_stats[f'rank_{rank}'] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': min(scores),
                    'max': max(scores)
                }
        
        analysis['rank_statistics'] = rank_stats
        
        return analysis
    
    def find_problematic_queries(self, queries: List[str], score_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Identifica queries com recuperação problemática."""
        print(f"\n🚨 Identificando queries problemáticas (threshold: {score_threshold})...")
        
        problematic = []
        
        for query in queries:
            results = search_similar_docs(query, k=5)
            
            if not results:
                problematic.append({
                    'query': query,
                    'issue': 'no_results',
                    'description': 'Nenhum resultado encontrado'
                })
                continue
            
            # Verificar se todos os scores são ruins
            scores = [score for doc, score in results]
            best_score = min(scores)  # Menor score = melhor similaridade
            
            if best_score > score_threshold:
                problematic.append({
                    'query': query,
                    'issue': 'poor_similarity',
                    'description': f'Melhor score: {best_score:.3f}',
                    'best_score': best_score,
                    'all_scores': scores
                })
            
            # Verificar se há muito pouca variação nos scores
            if len(scores) > 1:
                score_std = np.std(scores)
                if score_std < 0.01:  # Muito pouca variação
                    problematic.append({
                        'query': query,
                        'issue': 'low_discrimination',
                        'description': f'Baixa discriminação (std: {score_std:.4f})',
                        'score_std': score_std,
                        'scores': scores
                    })
        
        return problematic


def print_analysis_results(analysis: Dict[str, Any]):
    """Imprime resultados da análise de forma organizada."""
    print(f"\n📋 ANÁLISE DE RECUPERAÇÃO")
    print("="*60)
    
    if 'query' in analysis:
        print(f"🔍 Query: {analysis['query']}")
        
        if 'k_analysis' in analysis:
            print(f"\n📊 Análise por K:")
            for k, stats in analysis['k_analysis'].items():
                print(f"   K={k}: {stats['retrieved_count']} chunks, score médio: {stats['avg_score']:.3f}")
                print(f"        Fontes: {', '.join(stats['sources'])}")
        
        if 'chunk_details' in analysis and analysis['chunk_details']:
            print(f"\n📚 Detalhes dos chunks (Top 5):")
            for chunk in analysis['chunk_details']:
                print(f"   {chunk['rank']}. Score: {chunk['score']:.3f} | {chunk['source']}")
                print(f"      {chunk['content']}")


def main():
    """Script principal de análise de recuperação."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Análise de qualidade de recuperação")
    parser.add_argument("--query", type=str, help="Analisar uma query específica")
    parser.add_argument("--queries-file", type=str, help="Arquivo com queries para análise")
    parser.add_argument("--popularity", action="store_true", help="Analisar popularidade dos chunks")
    parser.add_argument("--scores", action="store_true", help="Analisar distribuição de scores")
    parser.add_argument("--problems", action="store_true", help="Encontrar queries problemáticas")
    parser.add_argument("--k", type=int, default=5, help="Número de chunks para recuperar")
    parser.add_argument("--threshold", type=float, default=0.8, help="Threshold para queries problemáticas")
    parser.add_argument("--all", action="store_true", help="Executar todas as análises")
    
    args = parser.parse_args()
    
    try:
        analyzer = RetrievalAnalyzer()
        
        # Queries padrão para teste
        default_queries = [
            "Como foi otimizado o processo de login?",
            "Qual tecnologia foi usada para cache?",
            "Quanto tempo levava o login antes?",
            "Que tipo de paralelismo foi implementado?",
            "Performance do sistema",
            "Otimização de APIs",
            "Cache distribuído",
            "Tempo de resposta"
        ]
        
        # Carregar queries de arquivo se especificado
        queries = default_queries
        if args.queries_file:
            try:
                with open(args.queries_file, 'r', encoding='utf-8') as f:
                    queries = [line.strip() for line in f if line.strip()]
                print(f"📂 Carregadas {len(queries)} queries do arquivo")
            except FileNotFoundError:
                print(f"❌ Arquivo não encontrado: {args.queries_file}")
        
        # Executar análises
        if args.query:
            analysis = analyzer.analyze_query_retrieval(args.query)
            print_analysis_results(analysis)
        
        if args.all or args.popularity:
            popularity = analyzer.analyze_chunk_popularity(queries, k=args.k)
            print(f"\n🏆 ANÁLISE DE POPULARIDADE")
            print(f"   Total de queries: {popularity['total_queries']}")
            print(f"   Total de recuperações: {popularity['total_retrievals']}")
            print(f"   Chunks únicos recuperados: {popularity['unique_chunks_retrieved']}")
            print(f"   Cobertura de chunks: {popularity['chunk_coverage']:.1%}")
            
            print(f"\n📚 Chunks mais populares:")
            for chunk_id, count in popularity['popular_chunks']:
                print(f"   {count}x: {chunk_id[:60]}...")
            
            print(f"\n📁 Fontes mais recuperadas:")
            for source, count in popularity['popular_sources']:
                print(f"   {count}x: {source}")
        
        if args.all or args.scores:
            scores_analysis = analyzer.analyze_score_distribution(queries, k=args.k)
            print(f"\n📈 ANÁLISE DE SCORES")
            print(f"   Total de scores: {scores_analysis['total_scores']}")
            print(f"   Score médio: {scores_analysis['mean_score']:.3f}")
            print(f"   Score mediano: {scores_analysis['median_score']:.3f}")
            print(f"   Desvio padrão: {scores_analysis['std_score']:.3f}")
            print(f"   Range: {scores_analysis['min_score']:.3f} - {scores_analysis['max_score']:.3f}")
            
            ranges = scores_analysis['score_ranges']
            print(f"\n🎯 Distribuição de qualidade:")
            print(f"   Excelente (≤0.3): {ranges['excellent']}")
            print(f"   Boa (0.3-0.5): {ranges['good']}")
            print(f"   Regular (0.5-0.7): {ranges['fair']}")
            print(f"   Ruim (>0.7): {ranges['poor']}")
        
        if args.all or args.problems:
            problems = analyzer.find_problematic_queries(queries, score_threshold=args.threshold)
            print(f"\n🚨 QUERIES PROBLEMÁTICAS")
            
            if problems:
                for problem in problems:
                    print(f"\n❌ {problem['query']}")
                    print(f"   Problema: {problem['issue']} - {problem['description']}")
                    if 'all_scores' in problem:
                        print(f"   Scores: {[f'{s:.3f}' for s in problem['all_scores']]}")
            else:
                print("   ✅ Nenhuma query problemática encontrada!")
        
        # Se nenhum argumento específico, mostrar ajuda
        if not any([args.query, args.queries_file, args.popularity, args.scores, args.problems, args.all]):
            print("🎯 Análise de Recuperação - Opções disponíveis:")
            print("   --query 'texto'     : Analisar query específica")
            print("   --queries-file FILE : Analisar queries de arquivo")
            print("   --popularity        : Analisar popularidade dos chunks")
            print("   --scores            : Analisar distribuição de scores")
            print("   --problems          : Encontrar queries problemáticas")
            print("   --all               : Executar todas as análises")
            print("   --k 5               : Número de chunks (padrão: 5)")
            print("   --threshold 0.8     : Threshold para problemas")
            print("\nExemplo: python scripts/analyze_retrieval.py --all")
        
    except Exception as e:
        print(f"❌ Erro durante análise: {e}")


if __name__ == "__main__":
    main()
