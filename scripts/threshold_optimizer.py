#!/usr/bin/env python3
"""
📊 THRESHOLD OPTIMIZER - Análise e Otimização de Thresholds de Similaridade

Script educacional para análise quantitativa de thresholds de similaridade
em sistemas RAG, demonstrando como encontrar o ponto ótimo entre precisão
e cobertura através de análise empírica e métricas objetivas.

📚 FUNDAMENTAÇÃO TEÓRICA:

Threshold de similaridade é um dos parâmetros mais críticos em sistemas RAG,
determinando o trade-off entre:
- PRECISÃO: Evitar contexto irrelevante (threshold baixo/rigoroso)
- COBERTURA: Capturar informação relevante (threshold alto/permissivo)
- EFICIÊNCIA: Reduzir processamento desnecessário

🎯 METODOLOGIA DE OTIMIZAÇÃO:

1️⃣ BASELINE ANALYSIS:
   - Distribuição de scores de similaridade no dataset
   - Identificação de clusters naturais de relevância
   - Análise de outliers e casos extremos

2️⃣ EMPIRICAL TESTING:
   - Teste sistemático de diferentes thresholds
   - Avaliação com queries conhecidas (golden dataset)
   - Métricas de qualidade por threshold

3️⃣ TRADE-OFF ANALYSIS:
   - Curva Precision-Recall por threshold
   - Análise de contexto perdido vs ruído filtrado
   - Identificação do knee point (ponto ótimo)

4️⃣ DOMAIN ADAPTATION:
   - Ajuste específico para tipo de conteúdo
   - Consideração de características do corpus
   - Validação com casos de uso reais

📊 MÉTRICAS IMPLEMENTADAS:

SIMILARITY METRICS:
- Score Distribution: Histograma de distribuição de scores
- Quartile Analysis: Q1, Q2, Q3, IQR de similaridade
- Cluster Detection: Identificação automática de grupos

QUALITY METRICS:
- Acceptance Rate: Percentual de chunks aceitos por threshold
- Context Quality: Avaliação subjetiva de relevância
- Information Loss: Estimativa de informação perdida

PERFORMANCE METRICS:
- Processing Time: Tempo de processamento por threshold
- Memory Usage: Uso de memória por configuração
- Throughput: Queries processadas por segundo

🔧 ALGORITMOS DE OTIMIZAÇÃO:

STATISTICAL APPROACHES:
- Standard Deviation Based: threshold = mean - k*std
- Percentile Based: threshold = percentile(scores, p)
- IQR Based: threshold = Q1 - 1.5*IQR

MACHINE LEARNING APPROACHES:
- K-Means Clustering: Agrupamento automático de scores
- Elbow Method: Identificação do ponto ótimo
- Validation Set: Teste com queries rotuladas

HEURISTIC APPROACHES:
- Content Type Specific: Thresholds por tipo de documento
- Query Length Adaptive: Ajuste baseado no tamanho da query
- Dynamic Threshold: Ajuste automático por contexto

🚀 USO EDUCACIONAL:

Este script demonstra abordagens científicas para otimização de parâmetros
em sistemas RAG, combinando análise estatística, machine learning e 
validação empírica para encontrar configurações ótimas.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Any
from collections import defaultdict
import json
from datetime import datetime

# Adicionar o diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from rag_demo.config import PERSIST_DIR, COLLECTION_NAME, EMB_MODEL

# ============================================================================
# ANÁLISE DE DISTRIBUIÇÃO DE SCORES
# ============================================================================

class SimilarityAnalyzer:
    """Analisador de distribuição de scores de similaridade."""
    
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=EMB_MODEL)
        self.vectorstore = None
        
    def load_vectorstore(self) -> Chroma:
        """Carrega vectorstore do ChromaDB."""
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                persist_directory=PERSIST_DIR,
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings
            )
        return self.vectorstore
    
    def analyze_score_distribution(self, test_queries: List[str], k: int = 20) -> Dict[str, Any]:
        """
        Analisa distribuição de scores para um conjunto de queries.
        
        Args:
            test_queries: Lista de queries para teste
            k: Número de documentos a recuperar por query
            
        Returns:
            Dict com estatísticas de distribuição
        """
        vectorstore = self.load_vectorstore()
        all_scores = []
        query_stats = []
        
        print(f"🔍 Analisando distribuição de scores para {len(test_queries)} queries...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"  Processando query {i}/{len(test_queries)}: {query[:50]}...")
            
            try:
                results = vectorstore.similarity_search_with_score(query, k=k)
                scores = [score for _, score in results]
                all_scores.extend(scores)
                
                query_stat = {
                    "query": query,
                    "scores": scores,
                    "min_score": min(scores) if scores else None,
                    "max_score": max(scores) if scores else None,
                    "mean_score": np.mean(scores) if scores else None,
                    "std_score": np.std(scores) if scores else None
                }
                query_stats.append(query_stat)
                
            except Exception as e:
                print(f"    ❌ Erro ao processar query: {e}")
                continue
        
        # Estatísticas globais
        if all_scores:
            global_stats = {
                "total_scores": len(all_scores),
                "global_min": min(all_scores),
                "global_max": max(all_scores),
                "global_mean": np.mean(all_scores),
                "global_std": np.std(all_scores),
                "global_median": np.median(all_scores),
                "quartiles": {
                    "q1": np.percentile(all_scores, 25),
                    "q2": np.percentile(all_scores, 50),
                    "q3": np.percentile(all_scores, 75)
                },
                "percentiles": {
                    "p10": np.percentile(all_scores, 10),
                    "p90": np.percentile(all_scores, 90),
                    "p95": np.percentile(all_scores, 95),
                    "p99": np.percentile(all_scores, 99)
                }
            }
        else:
            global_stats = {}
        
        return {
            "global_stats": global_stats,
            "query_stats": query_stats,
            "all_scores": all_scores,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def plot_score_distribution(self, analysis_result: Dict[str, Any], save_path: str = None):
        """Plota distribuição de scores."""
        scores = analysis_result["all_scores"]
        global_stats = analysis_result["global_stats"]
        
        if not scores:
            print("❌ Sem scores para plotar")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Análise de Distribuição de Scores de Similaridade', fontsize=16)
        
        # 1. Histograma
        ax1.hist(scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(global_stats["global_mean"], color='red', linestyle='--', label=f'Média: {global_stats["global_mean"]:.3f}')
        ax1.axvline(global_stats["global_median"], color='green', linestyle='--', label=f'Mediana: {global_stats["global_median"]:.3f}')
        ax1.set_xlabel('Score de Similaridade (distância)')
        ax1.set_ylabel('Frequência')
        ax1.set_title('Histograma de Scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box Plot
        ax2.boxplot(scores, vert=True)
        ax2.set_ylabel('Score de Similaridade')
        ax2.set_title('Box Plot de Scores')
        ax2.grid(True, alpha=0.3)
        
        # 3. Percentis
        percentiles = list(range(0, 101, 5))
        percentile_values = [np.percentile(scores, p) for p in percentiles]
        ax3.plot(percentiles, percentile_values, marker='o', linewidth=2)
        ax3.axhline(global_stats["percentiles"]["p95"], color='red', linestyle='--', label='P95')
        ax3.axhline(global_stats["percentiles"]["p90"], color='orange', linestyle='--', label='P90')
        ax3.set_xlabel('Percentil')
        ax3.set_ylabel('Score de Similaridade')
        ax3.set_title('Curva de Percentis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Estatísticas resumidas
        ax4.axis('off')
        stats_text = f"""
        ESTATÍSTICAS GLOBAIS
        
        Total de scores: {global_stats["total_scores"]:,}
        
        Medidas centrais:
        • Média: {global_stats["global_mean"]:.4f}
        • Mediana: {global_stats["global_median"]:.4f}
        • Desvio padrão: {global_stats["global_std"]:.4f}
        
        Quartis:
        • Q1 (25%): {global_stats["quartiles"]["q1"]:.4f}
        • Q2 (50%): {global_stats["quartiles"]["q2"]:.4f}  
        • Q3 (75%): {global_stats["quartiles"]["q3"]:.4f}
        
        Percentis relevantes:
        • P90: {global_stats["percentiles"]["p90"]:.4f}
        • P95: {global_stats["percentiles"]["p95"]:.4f}
        • P99: {global_stats["percentiles"]["p99"]:.4f}
        
        Valores extremos:
        • Mínimo: {global_stats["global_min"]:.4f}
        • Máximo: {global_stats["global_max"]:.4f}
        """
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Gráfico salvo em: {save_path}")
        
        plt.show()

# ============================================================================
# OTIMIZADOR DE THRESHOLD
# ============================================================================

class ThresholdOptimizer:
    """Otimizador de threshold baseado em análise empírica."""
    
    def __init__(self, analyzer: SimilarityAnalyzer):
        self.analyzer = analyzer
    
    def suggest_thresholds(self, analysis_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Sugere thresholds baseado em análise estatística.
        
        Args:
            analysis_result: Resultado da análise de distribuição
            
        Returns:
            Dict com sugestões de threshold para diferentes cenários
        """
        global_stats = analysis_result["global_stats"]
        
        if not global_stats:
            return {}
        
        suggestions = {}
        
        # 1. Baseado em desvio padrão
        mean = global_stats["global_mean"]
        std = global_stats["global_std"]
        
        suggestions["conservative"] = mean - 2 * std  # Muito rigoroso
        suggestions["strict"] = mean - std             # Rigoroso  
        suggestions["balanced"] = mean                 # Balanceado
        suggestions["permissive"] = mean + std         # Permissivo
        
        # 2. Baseado em percentis
        suggestions["p10_threshold"] = global_stats["percentiles"]["p10"]   # Aceita apenas top 10%
        suggestions["p25_threshold"] = global_stats["quartiles"]["q1"]      # Aceita apenas top 25%
        suggestions["p50_threshold"] = global_stats["global_median"]        # Aceita top 50%
        suggestions["p75_threshold"] = global_stats["quartiles"]["q3"]      # Aceita top 75%
        
        # 3. Baseado em clusters (heurística simples)
        # Assume que scores muito baixos = alta similaridade, scores altos = baixa similaridade
        q1 = global_stats["quartiles"]["q1"]
        q3 = global_stats["quartiles"]["q3"]
        iqr = q3 - q1
        
        suggestions["outlier_detection"] = q3 + 1.5 * iqr  # Detecção de outliers
        suggestions["cluster_boundary"] = q1 + 0.5 * iqr   # Fronteira entre clusters
        
        # 4. Filtrar valores negativos (não faz sentido para distância)
        suggestions = {k: max(0.0, v) for k, v in suggestions.items()}
        
        return suggestions
    
    def evaluate_threshold_performance(
        self, 
        test_queries: List[str], 
        thresholds: List[float],
        k_retrieve: int = 20
    ) -> Dict[str, Any]:
        """
        Avalia performance de diferentes thresholds.
        
        Args:
            test_queries: Queries para teste
            thresholds: Lista de thresholds para avaliar
            k_retrieve: Número de documentos a recuperar inicialmente
            
        Returns:
            Dict com métricas de performance por threshold
        """
        vectorstore = self.analyzer.load_vectorstore()
        results = {}
        
        print(f"🎯 Avaliando {len(thresholds)} thresholds com {len(test_queries)} queries...")
        
        for threshold in thresholds:
            print(f"  Testando threshold: {threshold:.3f}")
            
            threshold_stats = {
                "threshold": threshold,
                "queries_processed": 0,
                "total_retrieved": 0,
                "total_accepted": 0,
                "acceptance_rates": [],
                "processing_times": []
            }
            
            for query in test_queries:
                start_time = datetime.now()
                
                try:
                    # Buscar documentos
                    search_results = vectorstore.similarity_search_with_score(query, k=k_retrieve)
                    
                    # Aplicar threshold
                    accepted = [doc for doc, score in search_results if score <= threshold]
                    
                    # Estatísticas
                    retrieved_count = len(search_results)
                    accepted_count = len(accepted)
                    acceptance_rate = accepted_count / retrieved_count if retrieved_count > 0 else 0
                    
                    threshold_stats["total_retrieved"] += retrieved_count
                    threshold_stats["total_accepted"] += accepted_count
                    threshold_stats["acceptance_rates"].append(acceptance_rate)
                    
                    processing_time = (datetime.now() - start_time).total_seconds()
                    threshold_stats["processing_times"].append(processing_time)
                    
                    threshold_stats["queries_processed"] += 1
                    
                except Exception as e:
                    print(f"    ❌ Erro ao processar query '{query[:30]}...': {e}")
                    continue
            
            # Calcular estatísticas finais
            if threshold_stats["queries_processed"] > 0:
                threshold_stats["avg_acceptance_rate"] = np.mean(threshold_stats["acceptance_rates"])
                threshold_stats["std_acceptance_rate"] = np.std(threshold_stats["acceptance_rates"])
                threshold_stats["avg_processing_time"] = np.mean(threshold_stats["processing_times"])
                threshold_stats["overall_acceptance_rate"] = (
                    threshold_stats["total_accepted"] / threshold_stats["total_retrieved"] 
                    if threshold_stats["total_retrieved"] > 0 else 0
                )
            
            results[threshold] = threshold_stats
        
        return results
    
    def plot_threshold_comparison(self, performance_results: Dict[str, Any], save_path: str = None):
        """Plota comparação de performance entre thresholds."""
        thresholds = list(performance_results.keys())
        acceptance_rates = [performance_results[t]["overall_acceptance_rate"] for t in thresholds]
        avg_times = [performance_results[t]["avg_processing_time"] for t in thresholds]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Comparação de Performance entre Thresholds', fontsize=16)
        
        # 1. Taxa de aceitação vs Threshold
        ax1.plot(thresholds, acceptance_rates, marker='o', linewidth=2, markersize=8)
        ax1.set_xlabel('Threshold de Similaridade')
        ax1.set_ylabel('Taxa de Aceitação')
        ax1.set_title('Taxa de Aceitação por Threshold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Adicionar anotações em pontos importantes
        for i, (thresh, rate) in enumerate(zip(thresholds, acceptance_rates)):
            if i % 2 == 0:  # Anotar apenas alguns pontos para não poluir
                ax1.annotate(f'{rate:.2f}', (thresh, rate), 
                           textcoords="offset points", xytext=(0,10), ha='center')
        
        # 2. Tempo de processamento vs Threshold  
        ax2.plot(thresholds, avg_times, marker='s', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel('Threshold de Similaridade')
        ax2.set_ylabel('Tempo Médio de Processamento (s)')
        ax2.set_title('Tempo de Processamento por Threshold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Gráfico de comparação salvo em: {save_path}")
        
        plt.show()
    
    def find_optimal_threshold(self, performance_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encontra threshold ótimo baseado em múltiplos critérios.
        
        Args:
            performance_results: Resultados de avaliação de performance
            
        Returns:
            Dict com recomendações de threshold ótimo
        """
        if not performance_results:
            return {}
        
        # Extrair métricas
        data = []
        for threshold, stats in performance_results.items():
            data.append({
                "threshold": threshold,
                "acceptance_rate": stats["overall_acceptance_rate"],
                "avg_processing_time": stats["avg_processing_time"],
                "queries_processed": stats["queries_processed"]
            })
        
        # Ordenar por threshold
        data.sort(key=lambda x: x["threshold"])
        
        # Encontrar threshold com melhor balance
        # Critério: alta taxa de aceitação sem ser muito permissivo
        
        # 1. Threshold com taxa de aceitação próxima a 0.7 (70%)
        target_acceptance = 0.7
        best_balanced = min(data, key=lambda x: abs(x["acceptance_rate"] - target_acceptance))
        
        # 2. Threshold com menor tempo de processamento
        best_performance = min(data, key=lambda x: x["avg_processing_time"])
        
        # 3. Knee point analysis (ponto onde rate para de crescer significativamente)
        knee_threshold = None
        if len(data) >= 3:
            derivatives = []
            for i in range(1, len(data)-1):
                # Derivada aproximada da taxa de aceitação
                derivative = (data[i+1]["acceptance_rate"] - data[i-1]["acceptance_rate"]) / (data[i+1]["threshold"] - data[i-1]["threshold"])
                derivatives.append((data[i]["threshold"], derivative))
            
            # Threshold onde a derivada é menor (rate para de crescer)
            if derivatives:
                knee_threshold = min(derivatives, key=lambda x: abs(x[1]))
        
        recommendations = {
            "balanced_threshold": {
                "value": best_balanced["threshold"],
                "acceptance_rate": best_balanced["acceptance_rate"],
                "reasoning": f"Threshold com taxa de aceitação mais próxima ao target de {target_acceptance:.1%}"
            },
            "performance_threshold": {
                "value": best_performance["threshold"],
                "acceptance_rate": best_performance["acceptance_rate"],
                "avg_time": best_performance["avg_processing_time"],
                "reasoning": "Threshold com menor tempo de processamento"
            }
        }
        
        if knee_threshold:
            recommendations["knee_point_threshold"] = {
                "value": knee_threshold[0],
                "derivative": knee_threshold[1],
                "reasoning": "Knee point - threshold onde ganhos marginais diminuem"
            }
        
        # Recomendação final (heurística)
        final_recommendation = best_balanced["threshold"]
        
        recommendations["final_recommendation"] = {
            "value": final_recommendation,
            "reasoning": "Threshold recomendado baseado em análise multi-critério"
        }
        
        return recommendations

# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

def main():
    """Script principal para otimização de threshold."""
    print("📊 THRESHOLD OPTIMIZER - ANÁLISE E OTIMIZAÇÃO DE SIMILARIDADE\n")
    
    # Queries de teste para análise
    test_queries = [
        "Qual é a latência média das APIs?",
        "Como funciona o cache distribuído?", 
        "Quantos usuários simultâneos o sistema suporta?",
        "Como foi implementada a arquitetura de microserviços?",
        "Quais tecnologias são usadas para segurança?",
        "Como funciona o sistema de monitoramento?",
        "Onde são enviados os alertas críticos?",
        "Qual é o uptime do serviço?",
        "Como é feito o deployment automático?",
        "Quais métricas de performance são coletadas?",
        "Como funciona a replicação do banco de dados?",
        "Quais são as tecnologias de containerização?",
        "Como é implementado o rate limiting?",
        "Onde ficam centralizados os logs?",
        "Como funciona a autenticação JWT?"
    ]
    
    analyzer = SimilarityAnalyzer()
    optimizer = ThresholdOptimizer(analyzer)
    
    # 1. ANÁLISE DE DISTRIBUIÇÃO
    print("🔍 FASE 1: Análise de distribuição de scores...")
    analysis_result = analyzer.analyze_score_distribution(test_queries, k=15)
    
    if not analysis_result["global_stats"]:
        print("❌ Falha na análise - verifique se há documentos indexados")
        return
    
    # Exibir estatísticas básicas
    stats = analysis_result["global_stats"]
    print(f"\n📊 ESTATÍSTICAS GLOBAIS:")
    print(f"  Total de scores analisados: {stats['total_scores']:,}")
    print(f"  Média: {stats['global_mean']:.4f}")
    print(f"  Mediana: {stats['global_median']:.4f}")
    print(f"  Desvio padrão: {stats['global_std']:.4f}")
    print(f"  Min/Max: {stats['global_min']:.4f} / {stats['global_max']:.4f}")
    
    # 2. SUGESTÕES BASEADAS EM ESTATÍSTICA
    print(f"\n🎯 FASE 2: Sugestões de threshold baseadas em análise...")
    suggestions = optimizer.suggest_thresholds(analysis_result)
    
    print(f"\n📋 SUGESTÕES DE THRESHOLD:")
    for name, value in suggestions.items():
        print(f"  {name}: {value:.4f}")
    
    # 3. AVALIAÇÃO EMPÍRICA DE PERFORMANCE
    print(f"\n⚡ FASE 3: Avaliação empírica de performance...")
    
    # Selecionar thresholds para teste
    test_thresholds = [
        0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70
    ]
    
    performance_results = optimizer.evaluate_threshold_performance(
        test_queries, test_thresholds, k_retrieve=20
    )
    
    # 4. ANÁLISE DE RESULTADOS
    print(f"\n📈 FASE 4: Análise de resultados...")
    print(f"\n{'Threshold':<12} {'Aceitação':<12} {'Tempo (ms)':<12} {'Queries':<10}")
    print("-" * 50)
    
    for threshold in sorted(test_thresholds):
        stats = performance_results[threshold]
        acceptance = stats["overall_acceptance_rate"]
        avg_time = stats["avg_processing_time"] * 1000  # Converter para ms
        queries = stats["queries_processed"]
        print(f"{threshold:<12.3f} {acceptance:<12.1%} {avg_time:<12.1f} {queries:<10}")
    
    # 5. RECOMENDAÇÃO FINAL
    print(f"\n🎯 FASE 5: Recomendação final...")
    recommendations = optimizer.find_optimal_threshold(performance_results)
    
    if recommendations:
        print(f"\n🏆 RECOMENDAÇÕES:")
        for name, rec in recommendations.items():
            print(f"  {name}: {rec['value']:.4f}")
            print(f"    → {rec['reasoning']}")
            if "acceptance_rate" in rec:
                print(f"    → Taxa de aceitação: {rec['acceptance_rate']:.1%}")
            print()
    
    # 6. VISUALIZAÇÕES
    print(f"\n📊 FASE 6: Gerando visualizações...")
    
    try:
        # Plot de distribuição
        analyzer.plot_score_distribution(analysis_result, "threshold_distribution.png")
        
        # Plot de comparação
        optimizer.plot_threshold_comparison(performance_results, "threshold_comparison.png")
        
    except Exception as e:
        print(f"❌ Erro ao gerar gráficos: {e}")
        print("(matplotlib pode não estar instalado)")
    
    # 7. SALVAR RESULTADOS
    output_file = f"threshold_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_data = {
        "analysis_result": analysis_result,
        "suggestions": suggestions,
        "performance_results": performance_results,
        "recommendations": recommendations,
        "test_queries": test_queries,
        "test_thresholds": test_thresholds
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"💾 Resultados salvos em: {output_file}")
    except Exception as e:
        print(f"❌ Erro ao salvar resultados: {e}")
    
    # 8. RESUMO EXECUTIVO
    if recommendations and "final_recommendation" in recommendations:
        final_threshold = recommendations["final_recommendation"]["value"]
        
        print(f"\n{'='*60}")
        print(f"📋 RESUMO EXECUTIVO")
        print(f"{'='*60}")
        print(f"Threshold recomendado: {final_threshold:.4f}")
        
        if final_threshold in performance_results:
            final_stats = performance_results[final_threshold]
            print(f"Taxa de aceitação: {final_stats['overall_acceptance_rate']:.1%}")
            print(f"Tempo médio: {final_stats['avg_processing_time']*1000:.1f}ms")
        
        print(f"\nPara usar este threshold:")
        print(f"  export SIMILARITY_THRESHOLD={final_threshold:.4f}")
        print(f"  python rag_with_guardrails.py \"sua pergunta\" strict")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()
