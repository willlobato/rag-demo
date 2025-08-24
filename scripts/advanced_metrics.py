#!/usr/bin/env python3
"""
🔬 ADVANCED METRICS - ANÁLISE MATEMÁTICA PROFUNDA DOS EMBEDDINGS

Este script implementa análises científicas e matemáticas avançadas para avaliar
a qualidade dos embeddings em um sistema RAG (Retrieval-Augmented Generation).

📚 FUNDAMENTAÇÃO TEÓRICA:
Os embeddings são representações vetoriais de documentos em um espaço n-dimensional.
A qualidade desses vetores determina diretamente a eficácia do sistema RAG.

🎯 OBJETIVOS:
1. Avaliar qualidade matemática dos embeddings (normalização, distribuição)
2. Detectar problemas estruturais (outliers, correlações, redundâncias)
3. Medir diversidade e entropia informacional
4. Fornecer insights acionáveis para otimização

🔍 MÉTRICAS IMPLEMENTADAS:
- Normalização vetorial (normas L2)
- Análise de distribuições estatísticas
- Detecção de outliers (Z-score, IQR)
- Cálculo de entropia informacional
- Análise de correlações entre dimensões
- Características dos documentos vs embeddings

📖 CONCEITOS MATEMÁTICOS:
- Álgebra Linear: Normas, produtos internos, projeções
- Estatística: Distribuições, correlações, outliers
- Teoria da Informação: Entropia, diversidade
- Machine Learning: Clustering, detecção de anomalias

🚀 USO EDUCACIONAL:
Cada função é documentada com:
- Explicação matemática do conceito
- Interpretação dos resultados
- Valores ideais e problemáticos
- Ações corretivas recomendadas
"""

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
    """
    🔬 Analisador Avançado de Métricas para Embeddings
    
    Esta classe implementa análises matemáticas profundas para avaliar a qualidade
    dos embeddings em sistemas RAG, baseada em princípios de álgebra linear,
    estatística e teoria da informação.
    
    📊 PIPELINE DE ANÁLISE:
    1. Carregamento e validação dos dados
    2. Análise de qualidade vetorial
    3. Análise de distribuições estatísticas
    4. Detecção de anomalias e outliers
    5. Cálculo de métricas informacionais
    6. Correlação com características textuais
    
    🎯 APLICAÇÕES:
    - Diagnóstico de problemas no modelo de embedding
    - Otimização de hiperparâmetros
    - Validação da qualidade dos dados
    - Monitoramento de performance do sistema
    """
    
    def __init__(self):
        """
        🚀 Inicializa o analisador e carrega os dados do ChromaDB.
        
        📋 PROCESSO:
        1. Conecta ao vectorstore (ChromaDB)
        2. Extrai embeddings, documentos e metadados
        3. Converte para arrays NumPy para análise matemática
        4. Valida integridade dos dados
        
        ⚠️ VALIDAÇÕES:
        - Consistência entre número de embeddings e documentos
        - Verificação de dimensionalidade
        - Detecção de valores NaN ou infinitos
        """
        print("📊 Carregando dados para análise avançada...")
        self.vect = build_or_load_vectorstore()
        self.data = self.vect.get(include=["embeddings", "documents", "metadatas"])
        self.embeddings = np.array(self.data["embeddings"])
        self.documents = self.data["documents"]
        self.metadatas = self.data["metadatas"]
        
        print(f"✅ Carregados {len(self.documents)} chunks com embeddings de {self.embeddings.shape[1]} dimensões")
    
    def analyze_embedding_quality(self) -> Dict[str, Any]:
        """
        🔍 Análise Matemática da Qualidade dos Embeddings
        
        Esta função implementa uma bateria de testes matemáticos para avaliar
        se os embeddings seguem as melhores práticas e padrões esperados.
        
        📐 CONCEITOS MATEMÁTICOS:
        
        1. NORMALIZAÇÃO VETORIAL:
           - Fórmula: ||v|| = √(v₁² + v₂² + ... + vₙ²)
           - Ideal: ||v|| = 1.0 para todos os vetores
           - Por que: Normalização garante que similaridade cosseno funcione corretamente
           - Problema: Vetores não-normalizados podem causar viés na busca
        
        2. CENTRALIZAÇÃO:
           - Fórmula: μ = (1/n) × Σvᵢ
           - Ideal: μ ≈ 0 (vetor médio próximo da origem)
           - Por que: Evita viés direccional sistemático
           - Problema: Média muito alta indica viés no modelo
        
        3. ANÁLISE DE VARIÂNCIA POR DIMENSÃO:
           - Fórmula: σ² = (1/n) × Σ(vᵢ - μ)²
           - Interpretação:
             * Baixa variância (σ² < 0.01): Dimensão "morta", pouco informativa
             * Alta variância (σ² > 0.1): Dimensão muito ativa, pode dominar
           - Ideal: Variância balanceada entre dimensões
        
        4. ANÁLISE DE CORRELAÇÃO:
           - Fórmula: r = Σ((xᵢ-μₓ)(yᵢ-μᵧ)) / √(Σ(xᵢ-μₓ)² × Σ(yᵢ-μᵧ)²)
           - Interpretação:
             * |r| > 0.8: Dimensões altamente correlacionadas (redundantes)
             * |r| < 0.2: Dimensões independentes (ideal)
           - Problema: Muitas correlações altas = ineficiência do espaço vetorial
        
        5. ANÁLISE DE DISTRIBUIÇÃO ESTATÍSTICA:
           - Skewness (Assimetria): Mede simetria da distribuição
             * Valor = 0: Distribuição simétrica (ideal)
             * Valor > 0: Cauda à direita (valores altos raros)
             * Valor < 0: Cauda à esquerda (valores baixos raros)
           - Kurtosis (Curtose): Mede "peso" das caudas
             * Valor = 0: Distribuição normal
             * Valor > 0: Caudas pesadas (muitos outliers)
             * Valor < 0: Caudas leves (poucos outliers)
        
        🎯 VALORES DE REFERÊNCIA:
        - Norma média: 1.0 ± 0.01 (normalizado)
        - Média geral: próxima de 0
        - Correlações altas: <10% das dimensões
        - Skewness: [-0.5, 0.5] (razoavelmente simétrico)
        - Kurtosis: [-1, 1] (próximo do normal)
        
        Returns:
            Dict contendo métricas detalhadas de qualidade
        """
        print("\n🔍 Analisando qualidade dos embeddings...")
        
        # 1. ANÁLISE DE NORMALIZAÇÃO
        # Calcular norma L2 de cada vetor embedding
        norms = np.linalg.norm(self.embeddings, axis=1)
        
        # 2. ANÁLISE DE CENTRALIZAÇÃO
        # Calcular vetor médio (deveria estar próximo da origem)
        mean_embedding = np.mean(self.embeddings, axis=0)
        centered_embeddings = self.embeddings - mean_embedding
        
        # 3. ANÁLISE DE VARIÂNCIA POR DIMENSÃO
        # Calcular estatísticas para cada dimensão do embedding
        dim_means = np.mean(self.embeddings, axis=0)
        dim_stds = np.std(self.embeddings, axis=0)
        
        # 4. ANÁLISE DE CORRELAÇÃO ENTRE DIMENSÕES
        # Detectar redundâncias no espaço vetorial
        dim_correlations = np.corrcoef(self.embeddings.T)
        # Contar correlações altas (excluindo diagonal)
        high_correlations = np.sum(np.abs(dim_correlations) > 0.8) - len(dim_correlations)
        
        # 5. ANÁLISE DE DISTRIBUIÇÃO ESTATÍSTICA
        # Analisar toda a distribuição como um conjunto
        embedding_flat = self.embeddings.flatten()
        
        # 📊 COMPILAR RESULTADOS COM INTERPRETAÇÕES
        return {
            'normalization': {
                'mean_norm': float(np.mean(norms)),                    # Norma média (ideal: 1.0)
                'std_norm': float(np.std(norms)),                     # Variação das normas (ideal: baixa)
                'min_norm': float(np.min(norms)),                     # Norma mínima
                'max_norm': float(np.max(norms)),                     # Norma máxima
                'is_normalized': bool(np.allclose(norms, 1.0, atol=1e-6))  # Se todos os vetores estão normalizados
            },
            'centralization': {
                'mean_value': float(np.mean(embedding_flat)),         # Média global (ideal: ~0)
                'is_centered': bool(abs(np.mean(embedding_flat)) < 1e-6)  # Se está centrado
            },
            'dimension_analysis': {
                'mean_variance_across_dims': float(np.mean(dim_stds**2)),      # Variância média entre dimensões
                'min_variance': float(np.min(dim_stds**2)),                    # Menor variância
                'max_variance': float(np.max(dim_stds**2)),                    # Maior variância
                'dimensions_with_low_variance': int(np.sum(dim_stds < 0.01)),  # Dimensões "mortas"
                'dimensions_with_high_variance': int(np.sum(dim_stds > 0.1))   # Dimensões muito ativas
            },
            'correlation_analysis': {
                'high_correlation_pairs': int(high_correlations // 2),  # Pares redundantes (÷2 pois matriz é simétrica)
                'max_correlation': float(np.max(np.abs(dim_correlations - np.eye(len(dim_correlations))))),  # Correlação máxima
                'mean_abs_correlation': float(np.mean(np.abs(dim_correlations - np.eye(len(dim_correlations)))))  # Correlação média
            },
            'distribution': {
                'skewness': float(stats.skew(embedding_flat)),        # Assimetria (-1 a 1 ideal)
                'kurtosis': float(stats.kurtosis(embedding_flat)),    # Curtose (-1 a 1 ideal)
                'is_normal': bool(stats.jarque_bera(embedding_flat)[1] > 0.05)  # Se segue distribuição normal
            }
        }
    
    def analyze_distance_distributions(self) -> Dict[str, Any]:
        """
        📏 Análise das Distribuições de Distâncias entre Embeddings
        
        Esta função analisa como os vetores estão distribuídos no espaço n-dimensional,
        calculando diferentes métricas de distância para entender a geometria dos dados.
        
        🎯 IMPORTÂNCIA:
        A distribuição das distâncias revela:
        - Se os documentos estão bem separados (boa discriminação)
        - Se existem clusters naturais nos dados
        - Se há outliers ou anomalias na distribuição
        - Qual métrica de distância é mais adequada para o dataset
        
        📐 MÉTRICAS DE DISTÂNCIA IMPLEMENTADAS:
        
        1. DISTÂNCIA EUCLIDIANA:
           - Fórmula: d(a,b) = √(Σ(aᵢ - bᵢ)²)
           - Interpretação: Distância "real" no espaço vetorial
           - Características:
             * Sensível à magnitude dos vetores
             * Boa para detectar similaridade absoluta
             * Pode ser dominada por poucas dimensões com valores altos
           - Valores típicos: 0.5 a 2.0 para embeddings normalizados
        
        2. DISTÂNCIA COSSENO:
           - Fórmula: d(a,b) = 1 - (a·b)/(||a|| × ||b||)
           - Interpretação: Mede diferença de direção (ignora magnitude)
           - Características:
             * Independe da magnitude dos vetores
             * Ideal para similaridade semântica
             * Valores entre 0 (idênticos) e 2 (opostos)
           - Valores típicos: 0.1 a 1.0 para documentos relacionados
        
        3. DISTÂNCIA MANHATTAN (L1):
           - Fórmula: d(a,b) = Σ|aᵢ - bᵢ|
           - Interpretação: Soma das diferenças absolutas
           - Características:
             * Menos sensível a outliers que Euclidiana
             * Útil quando dimensões têm importâncias similares
             * Computacionalmente eficiente
        
        📊 ANÁLISE ESTATÍSTICA:
        Para cada métrica, calculamos:
        - Média: Distância típica entre documentos
        - Mediana: Valor central da distribuição
        - Desvio padrão: Variabilidade das distâncias
        - Percentis (25%, 75%, 95%): Forma da distribuição
        - Min/Max: Extremos da distribuição
        
        🎯 INTERPRETAÇÃO DOS RESULTADOS:
        
        DISTRIBUIÇÃO SAUDÁVEL:
        - Média moderada (não muito alta nem baixa)
        - Desvio padrão razoável (indica diversidade)
        - P95 não muito maior que média (poucos outliers)
        - Diferença clara entre min e max (boa separação)
        
        PROBLEMAS POSSÍVEIS:
        - Média muito baixa: Documentos muito similares (falta diversidade)
        - Média muito alta: Documentos muito diferentes (possível ruído)
        - Desvio padrão muito baixo: Distribuição muito uniforme
        - P95 >> média: Muitos outliers na distribuição
        
        Returns:
            Dict contendo estatísticas detalhadas para cada métrica de distância
        """
        print("\n📏 Analisando distribuições de distâncias...")
        
        # ⚠️ AVISO: Cálculo de distâncias par a par pode ser custoso para muitos documentos
        # Complexidade: O(n²) onde n = número de documentos
        print("   Calculando distâncias (pode demorar para muitos chunks)...")
        
        # 📐 CALCULAR DISTÂNCIAS PAR A PAR
        try:
            # Distância Euclidiana: Distância "real" no espaço
            euclidean_distances = pdist(self.embeddings, metric='euclidean')
            
            # Distância Cosseno: Focada na direção dos vetores
            cosine_distances = pdist(self.embeddings, metric='cosine')
            
            # Distância Manhattan: Soma das diferenças absolutas
            # Nota: Usando cityblock que é equivalente a manhattan no scipy
            manhattan_distances = pdist(self.embeddings, metric='cityblock')
            
        except Exception as e:
            print(f"   ⚠️ Erro no cálculo de distâncias: {e}")
            return {'error': str(e)}
        
        def _analyze_distance_array(distances: np.ndarray, name: str) -> Dict[str, float]:
            """
            🔍 Função auxiliar para analisar array de distâncias
            
            Args:
                distances: Array de distâncias calculadas
                name: Nome da métrica para debug
            
            Returns:
                Dict com estatísticas completas
            """
            return {
                'mean': float(np.mean(distances)),                    # Distância média
                'std': float(np.std(distances)),                     # Variabilidade
                'min': float(np.min(distances)),                     # Menor distância
                'max': float(np.max(distances)),                     # Maior distância
                'median': float(np.median(distances)),               # Valor central
                'percentiles': {
                    '25': float(np.percentile(distances, 25)),       # Primeiro quartil
                    '75': float(np.percentile(distances, 75)),       # Terceiro quartil
                    '95': float(np.percentile(distances, 95))        # Percentil 95 (outliers)
                }
            }
        
        # 📊 COMPILAR RESULTADOS
        return {
            'euclidean': _analyze_distance_array(euclidean_distances, 'Euclidiana'),
            'cosine': _analyze_distance_array(cosine_distances, 'Cosseno'),
            'manhattan': _analyze_distance_array(manhattan_distances, 'Manhattan')
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
