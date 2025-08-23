#!/usr/bin/env python3
# analyze_chunks.py
# Script para analisar chunks e seus overlaps

import sys
from pathlib import Path
from difflib import SequenceMatcher

# Adicionar o diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_demo.ingest import load_all_documents, create_chunks
from rag_demo.config import CHUNK_SIZE, CHUNK_OVERLAP


def highlight_overlap(text1, text2):
    """Destaca a sobreposição entre dois textos."""
    matcher = SequenceMatcher(None, text1, text2)
    match = matcher.find_longest_match(0, len(text1), 0, len(text2))
    
    if match.size > 0:
        overlap = text1[match.a:match.a + match.size]
        return overlap.strip()
    return ""


def analyze_chunk_overlaps(chunks):
    """Analisa overlaps entre chunks consecutivos."""
    print("=== ANÁLISE DE OVERLAPS ===\n")
    
    total_overlap_chars = 0
    overlaps_found = 0
    
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i].page_content
        next_chunk = chunks[i + 1].page_content
        
        # Verificar se são do mesmo documento
        current_source = chunks[i].metadata.get('source', '')
        next_source = chunks[i + 1].metadata.get('source', '')
        
        if current_source != next_source:
            print(f"Chunk {i+1} → Chunk {i+2}: Documentos diferentes")
            print(f"  {current_source} → {next_source}\n")
            continue
        
        # Encontrar overlap
        overlap = highlight_overlap(current_chunk, next_chunk)
        
        if overlap:
            overlaps_found += 1
            total_overlap_chars += len(overlap)
            
            print(f"Chunk {i+1} → Chunk {i+2}: {len(overlap)} caracteres sobrepostos")
            print(f"  Fonte: {current_source}")
            print(f"  Overlap: \"{overlap[:100]}{'...' if len(overlap) > 100 else ''}\"")
        else:
            print(f"Chunk {i+1} → Chunk {i+2}: Sem overlap detectado")
            print(f"  Fonte: {current_source}")
        
        print()
    
    return overlaps_found, total_overlap_chars


def show_chunk_stats(chunks):
    """Mostra estatísticas dos chunks."""
    print("=== ESTATÍSTICAS DE CHUNKING ===\n")
    
    sources = {}
    total_chars = 0
    chunk_sizes = []
    
    for i, chunk in enumerate(chunks):
        content = chunk.page_content
        source = chunk.metadata.get('source', 'desconhecido')
        
        # Estatísticas
        chunk_size = len(content)
        chunk_sizes.append(chunk_size)
        total_chars += chunk_size
        
        if source not in sources:
            sources[source] = {'count': 0, 'chars': 0}
        sources[source]['count'] += 1
        sources[source]['chars'] += chunk_size
    
    print(f"Configuração de chunking:")
    print(f"  CHUNK_SIZE: {CHUNK_SIZE}")
    print(f"  CHUNK_OVERLAP: {CHUNK_OVERLAP}")
    print()
    
    print(f"Total de chunks: {len(chunks)}")
    print(f"Total de caracteres: {total_chars:,}")
    print(f"Tamanho médio dos chunks: {total_chars / len(chunks):.1f} caracteres")
    print(f"Menor chunk: {min(chunk_sizes)} caracteres")
    print(f"Maior chunk: {max(chunk_sizes)} caracteres")
    print()
    
    print("Por fonte:")
    for source, stats in sources.items():
        print(f"  {source}: {stats['count']} chunks, {stats['chars']:,} caracteres")
    print()


def show_detailed_chunks(chunks, show_full=False):
    """Mostra cada chunk em detalhes."""
    print("=== CHUNKS DETALHADOS ===\n")
    
    for i, chunk in enumerate(chunks):
        content = chunk.page_content
        source = chunk.metadata.get('source', 'desconhecido')
        
        print(f"--- Chunk {i+1} ---")
        print(f"Fonte: {source}")
        print(f"Tamanho: {len(content)} caracteres")
        
        if show_full:
            print(f"Conteúdo completo:")
            print(f'"{content}"')
        else:
            print(f"Início: \"{content[:100]}{'...' if len(content) > 100 else ''}\"")
            print(f"Final:  \"{'...' if len(content) > 100 else ''}{content[-100:]}\"")
        
        print()


def main():
    """Script principal para análise de chunks."""
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("Uso: python analyze_chunks.py [--full]")
        print("  --full: Mostra o conteúdo completo de cada chunk")
        return
    
    show_full = len(sys.argv) > 1 and sys.argv[1] == '--full'
    
    try:
        print("[INFO] Carregando documentos...")
        documents = load_all_documents()
        
        print("[INFO] Criando chunks...")
        chunks = create_chunks(documents)
        
        print(f"[INFO] {len(chunks)} chunks criados\n")
        
        # Análises
        show_chunk_stats(chunks)
        
        overlaps_found, total_overlap_chars = analyze_chunk_overlaps(chunks)
        
        print(f"=== RESUMO DE OVERLAPS ===")
        print(f"Overlaps encontrados: {overlaps_found}")
        print(f"Total de caracteres sobrepostos: {total_overlap_chars:,}")
        if overlaps_found > 0:
            print(f"Média de overlap: {total_overlap_chars / overlaps_found:.1f} caracteres")
        print()
        
        show_detailed_chunks(chunks, show_full)
        
    except KeyboardInterrupt:
        print("\n[INFO] Análise interrompida pelo usuário.")
    except Exception as e:
        print(f"[ERROR] Erro durante a análise: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
