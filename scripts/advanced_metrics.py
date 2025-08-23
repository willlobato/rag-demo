#!/usr/bin/env python3
# advanced_metrics.py
# Script para métricas avançadas e insights matemáticos dos embeddings

import sys
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from scipy import stats
from scipy.spatial.distance import pdist, squareform

# Adicionar o diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_demo.rag import build_or_load_vectorstore


class AdvancedMetricsAnalyzer:
    """Analisador de métricas avançadas para embeddings."""
    
    def __init__(self):
        print("📊 Carregando dados para análise avançada...")
        self.vect = build_or_load_vectorstore()
        self.data = self.vect.get(include=["embeddings", "documents", "metadatas"])
        self.embeddings = np.array(self.data["embeddings"])
        self.documents = self.data["documents"]
        self.metadatas = self.data["metadatas"]
        
        print(f"✅ Carregados {len(self.documents)} chunks com embeddings de {self.embeddings.shape[1]} dimensões")
    
    def analyze_embedding_quality(self) -> Dict[str, Any]:
        """Analisa qualidade dos embeddings através de métricas matemáticas."""
        print("\n🔍 Analisando qualidade dos embeddings...")
        
        # Normalização - embeddings devem estar normalizados
        norms = np.linalg.norm(self.embeddings, axis=1)
        
        # Centralização
        mean_embedding = np.mean(self.embeddings, axis=0)
        centered_embeddings = self.embeddings - mean_embedding
        
        # Distribuição por dimensão
        dim_means = np.mean(self.embeddings, axis=0)
        dim_stds = np.std(self.embeddings, axis=0)
        
        # Correlação entre dimensões
        dim_correlations = np.corrcoef(self.embeddings.T)
        high_correlations = np.sum(np.abs(dim_correlations) > 0.8) - len(dim_correlations)  # Remove diagonal
        
        # Análise de distribuição
        embedding_flat = self.embeddings.flatten()
        
        return {
            'normalization': {
                'mean_norm': float(np.mean(norms)),
                'std_norm': float(np.std(norms)),
                'min_norm': float(np.min(norms)),
                'max_norm': float(np.max(norms)),
                'is_normalized': bool(np.allclose(norms, 1.0, atol=1e-6))
            },
            'centralization': {
                'mean_value': float(np.mean(embedding_flat)),
                'is_centered': bool(abs(np.mean(embedding_flat)) < 1e-6)
            },
            'dimension_analysis': {
                'mean_variance_across_dims': float(np.mean(dim_stds**2)),
                'min_variance': float(np.min(dim_stds**2)),
                'max_variance': float(np.max(dim_stds**2)),
                'dimensions_with_low_variance': int(np.sum(dim_stds < 0.01)),
                'dimensions_with_high_variance': int(np.sum(dim_stds > 0.1))
            },
            'correlation_analysis': {
                'high_correlation_pairs': int(high_correlations // 2),  # Dividir por 2 pois matriz é simétrica
                'max_correlation': float(np.max(np.abs(dim_correlations - np.eye(len(dim_correlations))))),
                'mean_abs_correlation': float(np.mean(np.abs(dim_correlations - np.eye(len(dim_correlations)))))
            },
            'distribution': {
                'skewness': float(stats.skew(embedding_flat)),
                'kurtosis': float(stats.kurtosis(embedding_flat)),
                'is_normal': bool(stats.jarque_bera(embedding_flat)[1] > 0.05)  # p-value > 0.05
            }
        }
    
    def analyze_distance_distributions(self) -> Dict[str, Any]:
        """Analisa distribuições de distâncias entre embeddings."""
        print("\n📏 Analisando distribuições de distâncias...")
        
        # Calcular distâncias par a par
        print("   Calculando distâncias (pode demorar para muitos chunks)...")
        
        # Euclidiana
        euclidean_distances = pdist(self.embeddings, metric='euclidean')
        
        # Cosseno
        cosine_distances = pdist(self.embeddings, metric='cosine')
        
        # Manhattan
        manhattan_distances = pdist(self.embeddings, metric='manhattan')
        
        return {
            'euclidean': {
                'mean': float(np.mean(euclidean_distances)),
                'std': float(np.std(euclidean_distances)),
                'min': float(np.min(euclidean_distances)),
                'max': float(np.max(euclidean_distances)),
                'median': float(np.median(euclidean_distances)),
                'percentiles': {
                    '25': float(np.percentile(euclidean_distances, 25)),
                    '75': float(np.percentile(euclidean_distances, 75)),
                    '95': float(np.percentile(euclidean_distances, 95))
                }
            },
            'cosine': {
                'mean': float(np.mean(cosine_distances)),
                'std': float(np.std(cosine_distances)),
                'min': float(np.min(cosine_distances)),
                'max': float(np.max(cosine_distances)),
                'median': float(np.median(cosine_distances)),
                'percentiles': {
                    '25': float(np.percentile(cosine_distances, 25)),
                    '75': float(np.percentile(cosine_distances, 75)),
                    '95': float(np.percentile(cosine_distances, 95))
                }
            },
            'manhattan': {
                'mean': float(np.mean(manhattan_distances)),
                'std': float(np.std(manhattan_distances)),
                'min': float(np.min(manhattan_distances)),
                'max': float(np.max(manhattan_distances)),
                'median': float(np.median(manhattan_distances)),
                'percentiles': {
                    '25': float(np.percentile(manhattan_distances, 25)),
                    '75': float(np.percentile(manhattan_distances, 75)),
                    '95': float(np.percentile(manhattan_distances, 95))
                }
            }
        }
    
    def analyze_document_characteristics(self) -> Dict[str, Any]:
        """Analisa características dos documentos e sua relação com embeddings."""
        print("\n📚 Analisando características dos documentos...")
        
        # Estatísticas de texto
        doc_lengths = [len(doc) for doc in self.documents]
        word_counts = [len(doc.split()) for doc in self.documents]
        
        # Agrupar por fonte
        source_stats = defaultdict(list)
        source_embeddings = defaultdict(list)
        
        for i, metadata in enumerate(self.metadatas):
            source = metadata.get('source', 'unknown')
            source_stats[source].append({
                'length': doc_lengths[i],
                'word_count': word_counts[i],
                'embedding_norm': np.linalg.norm(self.embeddings[i])
            })
            source_embeddings[source].append(self.embeddings[i])
        
        # Calcular estatísticas por fonte
        source_analysis = {}
        for source, stats in source_stats.items():
            lengths = [s['length'] for s in stats]
            word_counts_src = [s['word_count'] for s in stats]
            norms = [s['embedding_norm'] for s in stats]
            
            # Variabilidade dos embeddings dentro da fonte
            if len(source_embeddings[source]) > 1:
                src_embeddings = np.array(source_embeddings[source])
                intra_source_distances = pdist(src_embeddings, metric='cosine')
                intra_source_similarity = 1 - np.mean(intra_source_distances)
            else:
                intra_source_similarity = 1.0
            
            source_analysis[source] = {
                'document_count': len(stats),
                'avg_length': float(np.mean(lengths)),
                'avg_word_count': float(np.mean(word_counts_src)),
                'avg_embedding_norm': float(np.mean(norms)),
                'intra_source_similarity': float(intra_source_similarity)
            }
        
        return {
            'overall': {
                'total_documents': len(self.documents),
                'avg_doc_length': float(np.mean(doc_lengths)),
                'std_doc_length': float(np.std(doc_lengths)),
                'avg_word_count': float(np.mean(word_counts)),
                'std_word_count': float(np.std(word_counts)),
                'length_word_correlation': float(np.corrcoef(doc_lengths, word_counts)[0, 1])
            },
            'by_source': source_analysis
        }
    
    def detect_outliers(self, method: str = 'isolation_forest') -> Dict[str, Any]:
        """Detecta outliers nos embeddings."""
        print(f"\n🚨 Detectando outliers usando {method}...")
        
        outliers = {
            'method': method,
            'outlier_indices': [],
            'outlier_documents': [],
            'outlier_scores': []
        }
        
        if method == 'z_score':
            # Z-score baseado na distância ao centroide
            centroid = np.mean(self.embeddings, axis=0)
            distances = np.linalg.norm(self.embeddings - centroid, axis=1)
            z_scores = np.abs(stats.zscore(distances))
            
            threshold = 3.0
            outlier_mask = z_scores > threshold
            
            outliers['outlier_indices'] = np.where(outlier_mask)[0].tolist()
            outliers['outlier_scores'] = z_scores[outlier_mask].tolist()
        
        elif method == 'iqr':
            # IQR baseado na norma dos embeddings
            norms = np.linalg.norm(self.embeddings, axis=1)
            q1, q3 = np.percentile(norms, [25, 75])
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_mask = (norms < lower_bound) | (norms > upper_bound)
            outliers['outlier_indices'] = np.where(outlier_mask)[0].tolist()
            outliers['outlier_scores'] = norms[outlier_mask].tolist()
        
        # Adicionar documentos outliers
        for idx in outliers['outlier_indices']:
            outliers['outlier_documents'].append({
                'index': int(idx),
                'document': self.documents[idx][:200] + "..." if len(self.documents[idx]) > 200 else self.documents[idx],
                'source': self.metadatas[idx].get('source', 'unknown'),
                'length': len(self.documents[idx])
            })
        
        return outliers
    
    def calculate_embedding_entropy(self) -> Dict[str, float]:
        """Calcula entropia dos embeddings."""
        print("\n🌀 Calculando entropia dos embeddings...")
        
        # Discretizar embeddings para calcular entropia
        num_bins = 50
        
        # Entropia por dimensão
        dim_entropies = []
        for dim in range(self.embeddings.shape[1]):
            hist, _ = np.histogram(self.embeddings[:, dim], bins=num_bins)
            # Adicionar pequeno valor para evitar log(0)
            hist = hist + 1e-10
            prob = hist / np.sum(hist)
            entropy = -np.sum(prob * np.log2(prob))
            dim_entropies.append(entropy)
        
        # Entropia global (simplificada)
        all_values = self.embeddings.flatten()
        hist, _ = np.histogram(all_values, bins=num_bins)
        hist = hist + 1e-10
        prob = hist / np.sum(hist)
        global_entropy = -np.sum(prob * np.log2(prob))
        
        return {
            'global_entropy': float(global_entropy),
            'mean_dimension_entropy': float(np.mean(dim_entropies)),
            'std_dimension_entropy': float(np.std(dim_entropies)),
            'min_dimension_entropy': float(np.min(dim_entropies)),
            'max_dimension_entropy': float(np.max(dim_entropies))
        }


def print_metrics_results(results: Dict[str, Any]):
    """Imprime resultados das métricas de forma organizada."""
    print("\n📊 RELATÓRIO DE MÉTRICAS AVANÇADAS")
    print("=" * 80)
    
    # Qualidade dos embeddings
    if 'embedding_quality' in results:
        eq = results['embedding_quality']
        print(f"\n🔍 QUALIDADE DOS EMBEDDINGS:")
        
        norm = eq['normalization']
        print(f"   Normalização:")
        print(f"     Norma média: {norm['mean_norm']:.6f}")
        print(f"     Está normalizado: {'✅' if norm['is_normalized'] else '❌'}")
        
        dist = eq['distribution']
        print(f"   Distribuição:")
        print(f"     Assimetria: {dist['skewness']:.3f}")
        print(f"     Curtose: {dist['kurtosis']:.3f}")
        print(f"     É normal: {'✅' if dist['is_normal'] else '❌'}")
        
        corr = eq['correlation_analysis']
        print(f"   Correlações:")
        print(f"     Pares altamente correlacionados: {corr['high_correlation_pairs']}")
        print(f"     Correlação máxima: {corr['max_correlation']:.3f}")
    
    # Distâncias
    if 'distances' in results:
        dist = results['distances']
        print(f"\n📏 DISTRIBUIÇÕES DE DISTÂNCIA:")
        for metric, stats in dist.items():
            print(f"   {metric.upper()}:")
            print(f"     Média: {stats['mean']:.3f}")
            print(f"     Mediana: {stats['median']:.3f}")
            print(f"     Desvio padrão: {stats['std']:.3f}")
    
    # Características dos documentos
    if 'documents' in results:
        docs = results['documents']
        print(f"\n📚 CARACTERÍSTICAS DOS DOCUMENTOS:")
        overall = docs['overall']
        print(f"   Total: {overall['total_documents']}")
        print(f"   Tamanho médio: {overall['avg_doc_length']:.0f} caracteres")
        print(f"   Palavras médias: {overall['avg_word_count']:.0f}")
        
        print(f"\n   Por fonte:")
        for source, stats in docs['by_source'].items():
            print(f"     {source}:")
            print(f"       Documentos: {stats['document_count']}")
            print(f"       Similaridade interna: {stats['intra_source_similarity']:.3f}")
    
    # Outliers
    if 'outliers' in results:
        outliers = results['outliers']
        print(f"\n🚨 OUTLIERS ({outliers['method']}):")
        print(f"   Encontrados: {len(outliers['outlier_indices'])}")
        for doc in outliers['outlier_documents'][:3]:  # Mostrar apenas os primeiros 3
            print(f"     {doc['index']}: {doc['document'][:80]}...")
    
    # Entropia
    if 'entropy' in results:
        entropy = results['entropy']
        print(f"\n🌀 ENTROPIA:")
        print(f"   Global: {entropy['global_entropy']:.3f}")
        print(f"   Média por dimensão: {entropy['mean_dimension_entropy']:.3f}")
        print(f"   Desvio padrão: {entropy['std_dimension_entropy']:.3f}")


def main():
    """Script principal de métricas avançadas."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Métricas avançadas para embeddings")
    parser.add_argument("--quality", action="store_true", help="Analisar qualidade dos embeddings")
    parser.add_argument("--distances", action="store_true", help="Analisar distribuições de distância")
    parser.add_argument("--documents", action="store_true", help="Analisar características dos documentos")
    parser.add_argument("--outliers", type=str, choices=['z_score', 'iqr'], default='z_score', 
                       help="Detectar outliers")
    parser.add_argument("--entropy", action="store_true", help="Calcular entropia")
    parser.add_argument("--output", type=str, help="Salvar resultados em arquivo JSON")
    parser.add_argument("--all", action="store_true", help="Executar todas as análises")
    
    args = parser.parse_args()
    
    try:
        analyzer = AdvancedMetricsAnalyzer()
        
        if len(analyzer.documents) == 0:
            print("❌ Nenhum documento encontrado. Execute a ingestão primeiro.")
            return
        
        results = {}
        
        if args.all or args.quality:
            results['embedding_quality'] = analyzer.analyze_embedding_quality()
        
        if args.all or args.distances:
            results['distances'] = analyzer.analyze_distance_distributions()
        
        if args.all or args.documents:
            results['documents'] = analyzer.analyze_document_characteristics()
        
        if args.all or args.outliers:
            results['outliers'] = analyzer.detect_outliers(method=args.outliers)
        
        if args.all or args.entropy:
            results['entropy'] = analyzer.calculate_embedding_entropy()
        
        # Imprimir resultados
        if results:
            print_metrics_results(results)
            
            # Salvar em arquivo se especificado
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"\n💾 Resultados salvos em: {args.output}")
        else:
            print("🎯 Métricas Avançadas - Opções disponíveis:")
            print("   --all          : Executar todas as análises")
            print("   --quality      : Qualidade dos embeddings")
            print("   --distances    : Distribuições de distância")
            print("   --documents    : Características dos documentos")
            print("   --outliers     : Detectar outliers (z_score|iqr)")
            print("   --entropy      : Calcular entropia")
            print("   --output FILE  : Salvar resultados em JSON")
            print("\nExemplo: python scripts/advanced_metrics.py --all")
        
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        print("💡 Instale as dependências: pip install scipy")
    except Exception as e:
        print(f"❌ Erro durante análise: {e}")


if __name__ == "__main__":
    main()
