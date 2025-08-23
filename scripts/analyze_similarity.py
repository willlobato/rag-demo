#!/usr/bin/env python3
# analyze_similarity.py
# Script para análise de similaridade entre chunks

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
    """Cria heatmap de similaridade entre chunks."""
    print("🔥 Criando heatmap de similaridade...")
    
    # Calcular matriz de similaridade
    similarity_matrix = cosine_similarity(embeddings)
    
    # Criar labels truncados
    labels = [f"Chunk {i+1}: {text[:30]}..." for i, text in enumerate(texts)]
    
    # Configurar plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        annot=len(texts) <= 10,  # Mostrar valores apenas se poucos chunks
        fmt=".2f",
        cbar_kws={'label': 'Similaridade Cosseno'}
    )
    
    plt.title("Matriz de Similaridade entre Chunks")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Heatmap salvo em: {save_path}")
    
    return similarity_matrix


def find_duplicates_and_similar(embeddings, texts, threshold=0.9):
    """Encontra chunks duplicados ou muito similares."""
    print(f"🔍 Procurando chunks similares (threshold: {threshold})...")
    
    similarity_matrix = cosine_similarity(embeddings)
    duplicates = []
    
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            similarity = similarity_matrix[i][j]
            
            if similarity >= threshold:
                duplicates.append({
                    'chunk1_idx': i,
                    'chunk2_idx': j,
                    'similarity': similarity,
                    'text1': texts[i][:100] + "...",
                    'text2': texts[j][:100] + "..."
                })
    
    print(f"🎯 Encontrados {len(duplicates)} pares similares")
    
    for dup in duplicates:
        print(f"\n📊 Similaridade: {dup['similarity']:.3f}")
        print(f"   Chunk {dup['chunk1_idx'] + 1}: {dup['text1']}")
        print(f"   Chunk {dup['chunk2_idx'] + 1}: {dup['text2']}")
    
    return duplicates


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
