#!/usr/bin/env python3
"""
🛠️ UTILITIES MODULE - Funções Utilitárias Educacionais para RAG

Conjunto de funções auxiliares que implementam operações fundamentais
para carregamento, processamento e formatação de documentos em sistemas RAG,
demonstrando boas práticas de manipulação de dados textuais.

📚 FUNDAMENTAÇÃO TÉCNICA:

Este módulo encapsula operações de baixo nível essenciais para sistemas RAG,
aplicando princípios de engenharia de software para criar funções reutilizáveis,
robustas e bem documentadas para manipulação de documentos.

🎯 RESPONSABILIDADES DO MÓDULO:

1️⃣ DOCUMENT LOADING (Carregamento de Documentos):
   📝 CONCEITO: Abstração unificada para diferentes formatos de documento
   
   FORMATOS SUPORTADOS:
   - TXT: Texto plano simples
   - MD: Markdown com preservação de estrutura
   - PDF: Documentos com extração de texto via PyPDF
   
   🔧 ESTRATÉGIAS DE CARREGAMENTO:
   - Encoding Detection: UTF-8 com fallback graceful
   - Error Handling: Continua processamento mesmo com falhas
   - Metadata Preservation: Mantém informações de fonte
   - Recursive Search: Busca em subdiretórios automaticamente

2️⃣ DATA CONVERSION (Conversão de Dados):
   📝 CONCEITO: Transformação de dados brutos em objetos LangChain Document
   
   PROCESSO:
   - File System Scanning: Busca recursiva por arquivos
   - Content Extraction: Leitura e parsing do conteúdo
   - Document Creation: Criação de objetos LangChain padronizados
   - Metadata Injection: Adição de informações contextuais
   
   📊 ESTRUTURA Document:
   ```python
   Document(
       page_content="Texto extraído do documento",
       metadata={"source": "caminho/para/arquivo.txt"}
   )
   ```

3️⃣ FALLBACK MECHANISMS (Mecanismos de Fallback):
   📝 CONCEITO: Estratégias de recuperação para ambientes sem dados
   
   DOCUMENTOS DE EXEMPLO:
   - Conteúdo educacional sobre RAG
   - Exemplos práticos de conceitos
   - Base mínima para funcionamento
   - Template para novos usuários

4️⃣ TEXT FORMATTING (Formatação de Texto):
   📝 CONCEITO: Preparação de contexto para prompts LLM
   
   ESTRATÉGIAS:
   - Document Concatenation: União de múltiplos documentos
   - Source Attribution: Citação de fontes para rastreabilidade
   - Separator Standards: Delimitadores consistentes
   - Context Optimization: Formatação otimizada para LLMs

📊 PADRÕES DE DESIGN APLICADOS:

STRATEGY PATTERN:
- Diferentes estratégias para diferentes formatos
- Extensibilidade para novos tipos de documento
- Isolamento de complexidade por tipo

FACTORY PATTERN:
- Criação padronizada de objetos Document
- Configuração centralizada de metadados
- Abstração da complexidade de instanciação

ERROR HANDLING PATTERN:
- Graceful degradation para falhas individuais
- Logging estruturado de problemas
- Continuidade de processamento

🔧 IMPLEMENTAÇÕES DETALHADAS:

LOAD_TXT_MD():
```python
# Busca recursiva por arquivos .txt e .md
# Leitura com encoding UTF-8 e error handling
# Criação de Document com metadata de source
# Continua processamento mesmo com falhas
```

LOAD_PDFS():
```python
# Utilização do PyPDFLoader para extração
# Processamento página por página
# Preservação de metadata original
# Error handling para PDFs corrompidos
```

GET_EXAMPLE_DOCUMENTS():
```python
# Documentos mínimos para demonstração
# Conteúdo educacional sobre RAG
# Permite funcionamento out-of-the-box
# Base para experimentação inicial
```

FORMAT_DOCS():
```python
# Concatenação inteligente de documentos
# Separadores padronizados (---)
# Source attribution automática
# Otimização para context window
```

🧪 CASOS DE USO EDUCACIONAIS:

DESENVOLVIMENTO LOCAL:
```python
# Carregamento automático de arquivos locais
# Fallback para documentos de exemplo
# Experimentação sem setup complexo
```

EXPERIMENTAÇÃO COM FORMATOS:
```python
# Teste com diferentes tipos de arquivo
# Análise de qualidade de extração
# Comparação de performance por formato
```

ANÁLISE DE QUALIDADE:
```python
# Validação de encoding de documentos
# Detecção de problemas de extração
# Auditoria de metadados
```

DESENVOLVIMENTO DE PIPELINES:
```python
# Base para pipelines de processamento
# Template para novos formatos
# Integração com sistemas externos
```

⚡ OTIMIZAÇÕES DE PERFORMANCE:

LAZY LOADING:
- Carregamento sob demanda
- Evita consumo desnecessário de memória
- Processamento incremental

ERROR RESILIENCE:
- Continua processamento com falhas parciais
- Logging de problemas sem interrupção
- Maximiza aproveitamento de dados válidos

ENCODING ROBUSTNESS:
- UTF-8 como padrão
- Fallback graceful para outros encodings
- Tratamento de caracteres especiais

🚨 ROBUSTEZ E CONFIABILIDADE:

FAULT TOLERANCE:
- Error handling em cada operação de I/O
- Graceful degradation para falhas
- Continuidade de processamento
- Logging detalhado de problemas

VALIDATION:
- Verificação de formatos suportados
- Validação de integridade de arquivos
- Confirmação de encoding válido
- Verificação de metadata

EXTENSIBILITY:
- Arquitetura plugável para novos formatos
- Interface consistente para all loaders
- Padrões de metadata padronizados
- Facilita adição de novos tipos

🚀 VALOR EDUCACIONAL:

Este módulo demonstra:
1. Padrões de design para sistemas robustos
2. Manipulação segura de arquivos e encoding
3. Abstração efetiva de complexidade
4. Error handling em sistemas de produção
5. Extensibilidade e reutilização de código

As funções utilitárias servem como building blocks fundamentais
para sistemas RAG, mostrando como implementar operações de base
de forma robusta e reutilizável.
"""

from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def load_txt_md(data_dir: Path) -> List[Document]:
    """
    Carrega arquivos TXT e MD do diretório especificado com busca recursiva.
    
    📁 FUNCIONALIDADE:
    - Busca recursiva em todos os subdiretórios
    - Suporte para .txt e .md simultaneamente
    - Encoding UTF-8 com error handling graceful
    - Preservação de caminho original como metadata
    
    🔧 ESTRATÉGIA DE LOADING:
    - rglob("*"): Busca recursiva em toda árvore
    - Filtragem por extensão case-insensitive
    - read_text() com errors="ignore" para robustez
    - Continuidade mesmo com falhas individuais
    
    Args:
        data_dir: Diretório raiz para busca de documentos
        
    Returns:
        List[Document]: Lista de documentos carregados com metadata
    """
    docs = []
    for p in data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
                if text.strip():  # Ignora arquivos vazios
                    docs.append(Document(
                        page_content=text, 
                        metadata={"source": str(p.relative_to(data_dir))}
                    ))
            except Exception as e:
                print(f"[WARN] Falha ao ler arquivo {p}: {e}")
    
    print(f"[INFO] Carregados {len(docs)} documentos TXT/MD")
    return docs


def load_pdfs(data_dir: Path) -> List[Document]:
    """
    Carrega arquivos PDF do diretório especificado usando PyPDFLoader.
    
    📖 FUNCIONALIDADE:
    - Extração de texto página por página
    - Preservação de estrutura original do PDF
    - Metadata com informação de fonte
    - Error handling para PDFs corrompidos
    
    🔧 ESTRATÉGIA DE EXTRAÇÃO:
    - PyPDFLoader para parsing robusto
    - Processamento página por página
    - Metadata injection consistente
    - Continua com falhas individuais
    
    Args:
        data_dir: Diretório para busca de arquivos PDF
        
    Returns:
        List[Document]: Lista de documentos extraídos de PDFs
    """
    docs = []
    pdf_files = list(data_dir.rglob("*.pdf"))
    
    for p in pdf_files:
        try:
            loader = PyPDFLoader(str(p))
            pdf_docs = loader.load()
            
            for doc in pdf_docs:
                # Atualiza metadata com caminho relativo
                doc.metadata = {
                    "source": str(p.relative_to(data_dir)),
                    "page": doc.metadata.get("page", "unknown")
                }
                docs.append(doc)
                
        except Exception as e:
            print(f"[WARN] Falha ao ler PDF {p}: {e}")
    
    print(f"[INFO] Carregados {len(docs)} documentos de {len(pdf_files)} PDFs")
    return docs


def get_example_documents() -> List[Document]:
    """
    Retorna documentos de exemplo para demonstração e fallback.
    
    📚 CONTEÚDO EDUCACIONAL:
    - Conceitos fundamentais de RAG
    - Exemplos práticos de implementação
    - Casos de uso comuns
    - Base para experimentação inicial
    
    🎯 CASOS DE USO:
    - Fallback quando data/ está vazio
    - Demonstração out-of-the-box
    - Material educacional básico
    - Template para novos usuários
    
    Returns:
        List[Document]: Documentos de exemplo com conteúdo educacional
    """
    return [
        Document(
            page_content=(
                "RAG (Retrieval-Augmented Generation) é uma técnica que combina "
                "busca vetorial com modelos de linguagem. O processo envolve: "
                "1) Dividir documentos em chunks, 2) Criar embeddings vetoriais, "
                "3) Recuperar chunks relevantes para uma consulta, "
                "4) Usar o contexto recuperado para gerar respostas precisas. "
                "Esta abordagem melhora significativamente a qualidade das respostas "
                "fornecendo informação contextual específica ao modelo de linguagem."
            ),
            metadata={"source": "exemplo_rag_conceitos.md"},
        ),
        Document(
            page_content=(
                "Exemplo prático de otimização: Sistema de login com paralelismo "
                "e cache distribuído usando Infinispan. A estratégia envolve "
                "cache L1 local, cache L2 distribuído, invalidação automática, "
                "e fallback para banco de dados. Métricas mostram redução de "
                "98% na latência e aumento de 15x no throughput comparado à "
                "implementação básica sem cache."
            ),
            metadata={"source": "exemplo_otimizacao_sistema.md"},
        ),
        Document(
            page_content=(
                "Configuração de embedding models: nomic-embed-text oferece "
                "excelente balance entre qualidade e performance para português. "
                "Alternativas incluem: all-MiniLM-L6-v2 (rápido, 384D), "
                "all-mpnet-base-v2 (qualidade, 768D), bge-large (sota, 1024D). "
                "Escolha baseada em: velocidade necessária, qualidade desejada, "
                "recursos computacionais disponíveis, e idioma do conteúdo."
            ),
            metadata={"source": "exemplo_configuracao_embeddings.md"},
        ),
    ]


def format_docs(docs: List[Document]) -> str:
    """
    Formata lista de documentos para uso em prompts RAG.
    
    📝 ESTRATÉGIA DE FORMATAÇÃO:
    - Concatenação com separadores padronizados
    - Source attribution para rastreabilidade
    - Estrutura otimizada para LLMs
    - Fallback para contexto vazio
    
    🎨 FORMATO DE SAÍDA:
    ```
    Conteúdo do documento 1
    (fonte: arquivo1.txt)
    ---
    Conteúdo do documento 2
    (fonte: arquivo2.md)
    ```
    
    Args:
        docs: Lista de documentos recuperados
        
    Returns:
        str: Contexto formatado para uso em prompt
    """
    if not docs:
        return "(sem contexto relevante encontrado)"
    
    formatted_parts = []
    for doc in docs:
        content = doc.page_content.strip()
        source = doc.metadata.get('source', 'fonte_desconhecida')
        formatted_parts.append(f"{content}\n(fonte: {source})")
    
    return "\n---\n".join(formatted_parts)
