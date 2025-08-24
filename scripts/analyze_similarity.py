#!/usr/bin/env python3
"""
📈 ANALYZE SIMILARITY - ANÁLISE VISUAL E MATEMÁTICA DE SIMILARIDADES

Este script implementa análises visuais e matemáticas das similaridades entre chunks
de documentos em um sistema RAG, fornecendo insights sobre duplicatas, clusters e
padrões nos dados.

📚 FUNDAMENTAÇÃO TEÓRICA:
A similaridade entre documentos é medida principalmente através da similaridade cosseno
entre seus embeddings vetoriais. Esta métrica é fundamental para:
- Busca por similaridade (retrieval)
- Detecção de duplicatas
- Agrupamento temático (clustering)
- Avaliação da diversidade do dataset

🎯 OBJETIVOS:
1. Visualizar matriz de similaridades entre todos os chunks
2. Detectar documentos duplicados ou muito similares
3. Agrupar documentos por similaridade (clustering)
4. Analisar estatísticas dos embeddings
5. Identificar padrões e estruturas nos dados

🔍 ANÁLISES IMPLEMENTADAS:
- Heatmap de similaridade (visualização da matriz NxN)
- Detecção de duplicatas com threshold configurável
- Clustering K-means com visualização PCA
- Análise estatística dos embeddings
- Visualização da distribuição por dimensões

📐 CONCEITOS MATEMÁTICOS:
- Similaridade Cosseno: cos(θ) = (A·B)/(||A||×||B||)
- K-means Clustering: Minimização da distância intra-cluster
- PCA: Redução dimensional preservando máxima variância
- Estatísticas descritivas: Média, desvio padrão, normas L2

🎨 VISUALIZAÇÕES GERADAS:
- similarity_heatmap.png: Matriz de similaridades como heatmap
- clusters_plot.png: Visualização 2D dos clusters via PCA
- embedding_dimensions.png: Distribuição dos valores por dimensão

🚀 USO EDUCACIONAL:
Cada função é documentada com:
- Explicação matemática dos algoritmos
- Interpretação visual dos resultados
- Valores de threshold recomendados
- Identificação de problemas comuns
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Adicionar o diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_demo.rag import build_or_load_vectorstore


def get_embeddings_and_texts():
    """Obtém embeddings e textos de todos os chunks."""
    print("📚 Carregando dados do ChromaDB...")
    
    vect = build_or_load_vectorstore()
    data = vect.get(include=["embeddings", "documents", "metadatas"])
    
    embeddings = np.array(data["embeddings"])
    texts = data["documents"]
    metadatas = data["metadatas"]
    
    print(f"✅ Carregados {len(texts)} chunks com embeddings de {embeddings.shape[1]} dimensões")
    
    return embeddings, texts, metadatas


def create_similarity_heatmap(embeddings, texts, save_path="similarity_heatmap.png"):
    """
    🔥 Cria Heatmap Visual da Matriz de Similaridade
    
    Esta função gera uma visualização colorida da matriz de similaridades entre
    todos os pares de chunks, permitindo identificar visualmente padrões, 
    clusters e duplicatas nos dados.
    
    🎯 PROPÓSITO:
    - Visualização intuitiva das relações entre documentos
    - Identificação rápida de duplicatas (cores muito quentes)
    - Detecção de clusters temáticos (blocos de cores similares)
    - Avaliação da diversidade geral do dataset
    
    📐 CONCEITO MATEMÁTICO:
    Similaridade Cosseno: cos(θ) = (A·B) / (||A|| × ||B||)
    
    Onde:
    - A, B são vetores embedding dos documentos
    - ||A|| é a norma L2 do vetor A
    - Resultado varia de -1 a 1 (ou 0 a 1 para vetores normalizados)
    
    🎨 INTERPRETAÇÃO DO HEATMAP:
    - Verde escuro/Amarelo: Alta similaridade (0.8-1.0)
      → Documentos muito similares ou duplicatas
    - Verde médio: Similaridade moderada (0.5-0.8)
      → Documentos relacionados tematicamente
    - Verde claro: Baixa similaridade (0.2-0.5)
      → Documentos com alguma relação distante
    - Azul escuro: Muito diferentes (0.0-0.2)
      → Documentos de temas distintos
    
    🔍 PADRÕES A OBSERVAR:
    1. DIAGONAL PRINCIPAL: Sempre 1.0 (documento vs ele mesmo)
    2. BLOCOS QUENTES: Grupos de documentos similares (clusters)
    3. LINHAS/COLUNAS QUENTES: Documento similar a muitos outros
    4. MATRIZ UNIFORME: Todos documentos similares (falta diversidade)
    5. MATRIZ FRIA: Documentos muito diversos (boa separação)
    
    Args:
        embeddings (np.ndarray): Matriz de embeddings (n_docs × n_dims)
        texts (List[str]): Lista de textos correspondentes
        save_path (str): Caminho para salvar o heatmap
    
    Returns:
        np.ndarray: Matriz de similaridade calculada
    """
    print("🔥 Criando heatmap de similaridade...")
    
    # 📊 CALCULAR MATRIZ DE SIMILARIDADE
    # Retorna matriz NxN onde entry (i,j) = similaridade entre doc i e doc j
    similarity_matrix = cosine_similarity(embeddings)
    
    # 🏷️ CRIAR LABELS TRUNCADOS PARA VISUALIZAÇÃO
    # Limita texto a 30 caracteres para legibilidade
    labels = [f"Chunk {i+1}: {text[:30]}..." for i, text in enumerate(texts)]
    
    # 🎨 CONFIGURAR VISUALIZAÇÃO
    plt.figure(figsize=(12, 10))
    
    # 🌈 CRIAR HEATMAP COM SEABORN
    sns.heatmap(
        similarity_matrix,
        xticklabels=labels,           # Labels do eixo X
        yticklabels=labels,           # Labels do eixo Y
        cmap="viridis",               # Colormap: azul (baixo) → verde → amarelo (alto)
        annot=len(texts) <= 10,       # Mostrar valores numéricos se poucos chunks
        fmt=".2f",                    # Formato dos números (2 casas decimais)
        cbar_kws={'label': 'Similaridade Cosseno'}  # Label da barra de cores
    )
    
    # 📝 CONFIGURAR TÍTULOS E ROTAÇÕES
    plt.title("Matriz de Similaridade entre Chunks")
    plt.xticks(rotation=45, ha='right')  # Rotacionar labels X para legibilidade
    plt.yticks(rotation=0)               # Labels Y horizontais
    plt.tight_layout()                   # Ajustar layout automaticamente
    
    # 💾 SALVAR ARQUIVO COM ALTA QUALIDADE
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Heatmap salvo em: {save_path}")
    
    return similarity_matrix


def find_duplicates_and_similar(embeddings, texts, threshold=0.9):
    """
    🔍 Detecção Inteligente de Duplicatas e Documentos Similares
    
    Esta função implementa um algoritmo de detecção de duplicatas baseado em
    similaridade cosseno, permitindo identificar conteúdo redundante que pode
    afetar a qualidade do sistema RAG.
    
    🎯 IMPORTÂNCIA PARA RAG:
    1. EFICIÊNCIA: Duplicatas consomem espaço desnecessário
    2. QUALIDADE: Múltiplas cópias podem distorcer resultados de busca
    3. DIVERSIDADE: Reduzir redundância aumenta cobertura temática
    4. PERFORMANCE: Menos documentos = busca mais rápida
    
    📐 ALGORITMO:
    1. Calcular matriz de similaridade cosseno NxN
    2. Para cada par (i,j) onde i < j:
       - Se similarity(i,j) >= threshold: marcar como similar
    3. Ordenar por similaridade (maior primeiro)
    4. Retornar lista de pares similares com contexto
    
    🎚️ INTERPRETAÇÃO DOS THRESHOLDS:
    
    THRESHOLD = 0.95-1.0:
    - Duplicatas quase exatas
    - Mesmo conteúdo com pequenas variações
    - AÇÃO: Remover duplicatas
    
    THRESHOLD = 0.9-0.95:
    - Conteúdo muito similar
    - Mesma informação, palavras diferentes
    - AÇÃO: Avaliar necessidade, possivelmente consolidar
    
    THRESHOLD = 0.8-0.9:
    - Documentos relacionados
    - Mesmo tema, perspectivas diferentes
    - AÇÃO: Manter, mas monitorar
    
    THRESHOLD = 0.7-0.8:
    - Similaridade temática
    - Tópicos relacionados
    - AÇÃO: Útil para contexto, manter
    
    THRESHOLD < 0.7:
    - Similaridade baixa/coincidental
    - Documentos essencialmente diferentes
    - AÇÃO: Nenhuma, diversidade saudável
    
    🔍 RESULTADOS TÍPICOS:
    - 0 duplicatas: ✅ Dataset bem curado
    - 1-5% duplicatas: ⚠️ Aceitável, monitorar
    - 10-20% duplicatas: ❌ Problema de curadoria
    - >20% duplicatas: 🚨 Dataset precisa limpeza
    
    Args:
        embeddings (np.ndarray): Matriz de embeddings
        texts (List[str]): Textos correspondentes
        threshold (float): Limite de similaridade (0.0-1.0)
    
    Returns:
        List[Dict]: Lista de pares similares com metadados
    """
    print(f"🔍 Procurando chunks similares (threshold: {threshold})...")
    
    # 📊 CALCULAR MATRIZ DE SIMILARIDADE COMPLETA
    similarity_matrix = cosine_similarity(embeddings)
    duplicates = []
    
    # 🔄 PERCORRER TODAS AS COMBINAÇÕES DE PARES
    # Nota: i < j evita comparar documento consigo mesmo e duplicar pares
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            similarity = similarity_matrix[i][j]
            
            # ✅ SE SIMILARIDADE EXCEDE THRESHOLD, REGISTRAR
            if similarity >= threshold:
                duplicates.append({
                    'chunk1_idx': i,                                    # Índice do primeiro documento
                    'chunk2_idx': j,                                    # Índice do segundo documento
                    'similarity': similarity,                           # Score de similaridade
                    'text1': texts[i][:100] + "...",                   # Preview do primeiro texto
                    'text2': texts[j][:100] + "...",                   # Preview do segundo texto
                    'similarity_level': _classify_similarity(similarity)  # Classificação textual
                })
    
    # 📊 ORDENAR POR SIMILARIDADE (MAIOR PRIMEIRO)
    duplicates.sort(key=lambda x: x['similarity'], reverse=True)
    
    # 📈 ESTATÍSTICAS RESUMO
    print(f"🎯 Encontrados {len(duplicates)} pares similares")
    
    if duplicates:
        avg_similarity = np.mean([d['similarity'] for d in duplicates])
        max_similarity = duplicates[0]['similarity']
        print(f"   📊 Similaridade média: {avg_similarity:.3f}")
        print(f"   📈 Similaridade máxima: {max_similarity:.3f}")
    
    # 📝 MOSTRAR EXEMPLOS DOS PARES MAIS SIMILARES
    for i, dup in enumerate(duplicates[:5]):  # Mostrar apenas os top 5
        print(f"\n📊 Par #{i+1} - Similaridade: {dup['similarity']:.3f} ({dup['similarity_level']})")
        print(f"   Chunk {dup['chunk1_idx'] + 1}: {dup['text1']}")
        print(f"   Chunk {dup['chunk2_idx'] + 1}: {dup['text2']}")
    
    if len(duplicates) > 5:
        print(f"\n   ... e mais {len(duplicates) - 5} pares similares")
    
    return duplicates


def _classify_similarity(similarity: float) -> str:
    """
    🏷️ Classifica nível de similaridade em categorias textuais
    
    Args:
        similarity (float): Score de similaridade (0.0-1.0)
    
    Returns:
        str: Classificação textual do nível
    """
    if similarity >= 0.95:
        return "DUPLICATA QUASE EXATA"
    elif similarity >= 0.9:
        return "MUITO SIMILAR"
    elif similarity >= 0.8:
        return "SIMILAR"
    elif similarity >= 0.7:
        return "RELACIONADO"
    else:
        return "POUCO SIMILAR"


def cluster_documents(embeddings, texts, n_clusters=3, save_path="clusters_plot.png"):
    """Agrupa documentos em clusters por similaridade."""
    print(f"🎭 Criando {n_clusters} clusters de documentos...")
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # PCA para visualização 2D
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Plot dos clusters
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c=cluster_labels, 
        cmap='tab10',
        alpha=0.7,
        s=100
    )
    
    # Adicionar labels aos pontos
    for i, (x, y) in enumerate(embeddings_2d):
        plt.annotate(
            f'C{i+1}', 
            (x, y), 
            xytext=(5, 5), 
            textcoords='offset points',
            fontsize=8,
            alpha=0.8
        )
    
    plt.colorbar(scatter, label='Cluster')
    plt.title('Clustering de Documentos (PCA 2D)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} da variância)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} da variância)')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Plot de clusters salvo em: {save_path}")
    
    # Analisar clusters
    print(f"\n🔍 Análise dos clusters:")
    for cluster_id in range(n_clusters):
        cluster_texts = [texts[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
        print(f"\n📁 Cluster {cluster_id + 1} ({len(cluster_texts)} chunks):")
        
        for i, text in enumerate(cluster_texts):
            print(f"   {i+1}. {text[:80]}...")
    
    return cluster_labels, embeddings_2d


def analyze_embedding_statistics(embeddings):
    """Analisa estatísticas dos embeddings."""
    print("📊 Analisando estatísticas dos embeddings...")
    
    # Estatísticas básicas
    print(f"\n📈 Estatísticas dos Embeddings:")
    print(f"   Forma: {embeddings.shape}")
    print(f"   Média geral: {np.mean(embeddings):.6f}")
    print(f"   Desvio padrão geral: {np.std(embeddings):.6f}")
    print(f"   Valor mínimo: {np.min(embeddings):.6f}")
    print(f"   Valor máximo: {np.max(embeddings):.6f}")
    
    # Normas L2
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\n📏 Normas L2:")
    print(f"   Média: {np.mean(norms):.6f}")
    print(f"   Desvio padrão: {np.std(norms):.6f}")
    print(f"   Mínima: {np.min(norms):.6f}")
    print(f"   Máxima: {np.max(norms):.6f}")
    
    # Similaridade média
    similarity_matrix = cosine_similarity(embeddings)
    # Remover diagonal (auto-similaridade)
    np.fill_diagonal(similarity_matrix, 0)
    avg_similarity = np.mean(similarity_matrix)
    
    print(f"\n🎯 Similaridade Cosseno:")
    print(f"   Similaridade média entre chunks: {avg_similarity:.6f}")
    print(f"   Similaridade máxima: {np.max(similarity_matrix):.6f}")
    print(f"   Similaridade mínima: {np.min(similarity_matrix):.6f}")


def visualize_embedding_dimensions(embeddings, save_path="embedding_dimensions.png"):
    """Visualiza distribuição das dimensões dos embeddings."""
    print("📊 Criando visualização das dimensões...")
    
    # Estatísticas por dimensão
    dim_means = np.mean(embeddings, axis=0)
    dim_stds = np.std(embeddings, axis=0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot das médias por dimensão
    ax1.plot(dim_means, alpha=0.7)
    ax1.set_title('Média dos Valores por Dimensão')
    ax1.set_xlabel('Dimensão')
    ax1.set_ylabel('Valor Médio')
    ax1.grid(True, alpha=0.3)
    
    # Plot dos desvios padrão por dimensão
    ax2.plot(dim_stds, alpha=0.7, color='orange')
    ax2.set_title('Desvio Padrão por Dimensão')
    ax2.set_xlabel('Dimensão')
    ax2.set_ylabel('Desvio Padrão')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Visualização salva em: {save_path}")


def main():
    """Script principal de análise de similaridade."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Análise de similaridade entre chunks")
    parser.add_argument("--heatmap", action="store_true", help="Criar heatmap de similaridade")
    parser.add_argument("--duplicates", type=float, default=0.9, help="Threshold para detectar duplicatas")
    parser.add_argument("--clusters", type=int, default=3, help="Número de clusters")
    parser.add_argument("--stats", action="store_true", help="Mostrar estatísticas dos embeddings")
    parser.add_argument("--dimensions", action="store_true", help="Visualizar dimensões dos embeddings")
    parser.add_argument("--all", action="store_true", help="Executar todas as análises")
    
    args = parser.parse_args()
    
    try:
        # Carregar dados
        embeddings, texts, metadatas = get_embeddings_and_texts()
        
        if len(texts) == 0:
            print("❌ Nenhum chunk encontrado. Execute a ingestão primeiro.")
            return
        
        # Executar análises baseadas nos argumentos
        if args.all or args.stats:
            analyze_embedding_statistics(embeddings)
        
        if args.all or args.heatmap:
            create_similarity_heatmap(embeddings, texts)
        
        if args.all or args.duplicates:
            find_duplicates_and_similar(embeddings, texts, threshold=args.duplicates)
        
        if args.all or args.clusters:
            cluster_documents(embeddings, texts, n_clusters=args.clusters)
        
        if args.all or args.dimensions:
            visualize_embedding_dimensions(embeddings)
        
        # Se nenhum argumento específico, mostrar ajuda
        if not any([args.heatmap, args.duplicates, args.clusters, args.stats, args.dimensions, args.all]):
            print("🎯 Análise de Similaridade - Opções disponíveis:")
            print("   --all           : Executar todas as análises")
            print("   --stats         : Estatísticas dos embeddings")
            print("   --heatmap       : Criar heatmap de similaridade")
            print("   --duplicates 0.9: Encontrar chunks similares (threshold)")
            print("   --clusters 3    : Agrupar em clusters")
            print("   --dimensions    : Visualizar dimensões")
            print("\nExemplo: python scripts/analyze_similarity.py --all")
        
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        print("💡 Instale as dependências: pip install matplotlib seaborn scikit-learn")
    except Exception as e:
        print(f"❌ Erro durante análise: {e}")


if __name__ == "__main__":
    main()
