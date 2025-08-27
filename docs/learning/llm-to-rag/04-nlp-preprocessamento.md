# Capítulo 04 — NLP e Pré-processamento: Preparando o Texto para o RAG

Nos capítulos anteriores, aprendemos sobre LLMs, embeddings e bases vetoriais. Mas há uma etapa crucial que acontece antes de gerar embeddings: **o pré-processamento do texto**. Como dividir documentos longos em pedaços menores? Como garantir que cada chunk tenha contexto suficiente? Como lidar com diferentes formatos de arquivo?

Neste capítulo, vamos mergulhar no **processamento de linguagem natural (NLP)** aplicado ao RAG, focando nas técnicas que transformam documentos brutos em chunks prontos para indexação.

**Resumo:** neste capítulo exploramos técnicas de NLP para pré-processar texto, estratégias de chunking, normalização e como preparar documentos de diferentes formatos para o pipeline RAG.

**Sumário:**
1. Por que o pré-processamento importa no RAG?
2. Estratégias de chunking (divisão de texto)
3. Normalização e limpeza de texto
4. Lidando com diferentes formatos de arquivo
5. Preservando contexto e metadados
6. Implementação prática no projeto
7. Otimizações avançadas
8. Limitações e trade-offs
9. Conclusão

---

## 1. Por que o pré-processamento importa no RAG?

Imagine que você tem um manual técnico de 200 páginas. Se você gerar um único embedding para todo o documento, ele será muito genérico. Se você gerar um embedding para cada palavra, perderá o contexto. O **chunking** é o meio-termo: dividir o texto em pedaços que mantêm significado coerente.

Problemas comuns sem pré-processamento adequado:

- **Chunks muito grandes:** embeddings genéricos, menos precisão na busca.
- **Chunks muito pequenos:** perda de contexto, fragmentação de ideias.
- **Texto sujo:** caracteres especiais, formatação que atrapalha a geração de embeddings.
- **Falta de metadados:** impossível rastrear a origem dos chunks recuperados.

O pré-processamento resolve esses problemas, garantindo que cada chunk seja **semanticamente coerente** e **tecnicamente otimizado**.

---

## 2. Estratégias de chunking (divisão de texto)

### 2.1. Chunking por tamanho fixo

A abordagem mais simples: dividir texto em pedaços de N caracteres ou tokens.

```python
def chunk_by_size(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Overlap para preservar contexto
    return chunks
```

**Vantagens:** simples, previsível.
**Desvantagens:** pode quebrar frases no meio, perder contexto semântico.

### 2.2. Chunking recursivo (usado no projeto)

O `RecursiveCharacterTextSplitter` do LangChain tenta dividir por separadores naturais:

1. Primeiro, tenta dividir por parágrafos (`\n\n`).
2. Se o chunk ainda for muito grande, divide por frases (`.`, `!`, `?`).
3. Se ainda for grande, divide por palavras (` `).
4. Como último recurso, divide por caracteres.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_text(document_text)
```

**Vantagens:** preserva estrutura natural do texto.
**Desvantagens:** chunks de tamanho variável.

### 2.3. Chunking semântico

Divide texto baseado em mudanças de tópico ou estrutura semântica.

```python
# Exemplo conceitual - requer análise mais sofisticada
def semantic_chunking(text):
    sentences = split_into_sentences(text)
    current_chunk = []
    chunks = []
    
    for sentence in sentences:
        if topic_changed(current_chunk, sentence):
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
        else:
            current_chunk.append(sentence)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks
```

**Vantagens:** chunks semanticamente coerentes.
**Desvantagens:** complexo, computacionalmente caro.

---

## 3. Normalização e limpeza de texto

Antes do chunking, é uma boa prática limpar e normalizar o texto para melhorar a qualidade dos embeddings.

- **Remover caracteres estranhos:** `\u0000`, `\ufffd`, etc.
- **Normalizar espaços em branco:** substituir múltiplos espaços/quebras de linha por um único.
- **Converter para minúsculas (opcional):** pode ajudar em alguns modelos de embedding, mas pode prejudicar em outros que são sensíveis a maiúsculas (ex: siglas).
- **Remover cabeçalhos/rodapés repetitivos:** em PDFs ou documentos corporativos.

```python
import re

def normalize_text(text):
    text = text.lower()  # Cuidado com siglas
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[\n\r]+', '\n', text)
    return text
```

### Técnicas Adicionais de Limpeza

- **Remoção de HTML:** Se seus documentos vêm de fontes web, é crucial remover tags HTML para não poluir os embeddings. Bibliotecas como `BeautifulSoup` são excelentes para isso.
  ```python
  from bs4 import BeautifulSoup

  def strip_html(text):
      soup = BeautifulSoup(text, "html.parser")
      return soup.get_text()
  ```

- **Tratamento de URLs e E-mails:** URLs e endereços de e-mail raramente adicionam valor semântico e podem ser substituídos por um placeholder ou removidos.
  ```python
  def remove_urls_emails(text):
      text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
      text = re.sub(r'\S*@\S*\s?', '', text, flags=re.MULTILINE)
      return text
  ```

- **Lidar com Hifenação:** Palavras que são quebradas por hífens no final de uma linha (`quebra-\nde-linha`) devem ser unidas para formar a palavra correta (`quebra-de-linha`).

- **Expansão de Contrações (em inglês):** Em textos em inglês, expandir contrações como "don't" para "do not" pode ajudar na consistência.

A escolha das técnicas de limpeza depende muito da natureza dos seus dados. A regra geral é: quanto mais "limpo" e semanticamente puro for o texto, melhor será o desempenho do seu sistema RAG.

---

## 4. Lidando com diferentes formatos de arquivo

### 4.1. Arquivos de texto (.txt, .md)

```python
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return clean_text(content)
```

### 4.2. Arquivos PDF

```python
import fitz  # PyMuPDF

def load_pdf_file(file_path):
    doc = fitz.open(file_path)
    text = ""
    
    for page in doc:
        text += page.get_text()
        text += "\n\n"  # Separar páginas
    
    doc.close()
    return clean_text(text)
```

### 4.3. Arquivos Word (.docx)

```python
from docx import Document

def load_docx_file(file_path):
    doc = Document(file_path)
    text = ""
    
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    
    return clean_text(text)
```

---

## 5. Preservando contexto e metadados

### 5.1. Metadados essenciais

Cada chunk deve manter informações sobre sua origem:

```python
class DocumentChunk:
    def __init__(self, content, metadata):
        self.content = content
        self.metadata = metadata
        
# Exemplo de metadados
metadata = {
    "source": "manual_infinispan.pdf",
    "page": 15,
    "chapter": "Cache Configuration",
    "chunk_id": "manual_infinispan_chunk_23",
    "created_at": "2024-08-24T10:30:00Z"
}
```

### 5.2. Preservando contexto entre chunks

```python
def create_chunks_with_context(text, chunk_size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    
    chunks = splitter.split_text(text)
    
    # Adiciona contexto do chunk anterior
    enhanced_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            # Adiciona últimas frases do chunk anterior
            prev_context = get_last_sentences(chunks[i-1], max_sentences=2)
            chunk = f"[Contexto anterior: {prev_context}]\n\n{chunk}"
        enhanced_chunks.append(chunk)
    
    return enhanced_chunks
```

---

## 6. Implementação prática no projeto

No nosso projeto, o pré-processamento acontece em `rag_demo/ingest.py`:

### 6.1. Carregamento de documentos

```python
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader
)

def load_documents(data_dir="data"):
    loaders = [
        DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader(data_dir, glob="**/*.md", loader_cls=TextLoader),
        DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader),
    ]
    
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    
    return documents
```

### 6.2. Chunking configurável

```python
def create_text_splitter(chunk_size=1000, chunk_overlap=200):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

def process_documents(documents):
    text_splitter = create_text_splitter()
    chunks = text_splitter.split_documents(documents)
    
    # Adiciona metadados extras
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i}"
        chunk.metadata["processed_at"] = datetime.now().isoformat()
    
    return chunks
```

### 6.3. Exemplo de uso

```bash
# Adicione arquivos em data/
echo "Infinispan é um cache distribuído..." > data/infinispan.txt
echo "Redis é um banco de dados em memória..." > data/redis.txt

# Execute o processamento
python scripts/run_ingest.py

# Verifique os chunks gerados
python scripts/analyze_chunks.py
```

---

## 7. Otimizações avançadas

### 7.1. Chunking adaptativo por tipo de documento

```python
def get_optimal_chunk_size(document_type, content_length):
    if document_type == "manual":
        return min(1500, content_length // 10)
    elif document_type == "api_doc":
        return 800
    elif document_type == "faq":
        return 500
    else:
        return 1000
```

### 7.2. Filtragem de chunks de baixa qualidade

```python
def filter_low_quality_chunks(chunks):
    filtered = []
    for chunk in chunks:
        # Remove chunks muito pequenos
        if len(chunk.page_content) < 50:
            continue
            
        # Remove chunks que são só números/símbolos
        if not re.search(r'[a-zA-Z]{10,}', chunk.page_content):
            continue
            
        # Remove chunks duplicados
        if chunk.page_content not in [c.page_content for c in filtered]:
            filtered.append(chunk)
    
    return filtered
```

### 7.3. Análise de qualidade dos chunks

```python
def analyze_chunk_quality(chunks):
    stats = {
        "total_chunks": len(chunks),
        "avg_length": sum(len(c.page_content) for c in chunks) / len(chunks),
        "unique_sources": len(set(c.metadata.get("source") for c in chunks))
    }
    
    print(f"Análise dos chunks:")
    print(f"Total: {stats['total_chunks']}")
    print(f"Tamanho médio: {stats['avg_length']:.0f} caracteres")
    print(f"Fontes únicas: {stats['unique_sources']}")
    
    return stats
```

---

## 8. Limitações e trade-offs

### 8.1. Tamanho do chunk vs. precisão

- **Chunks maiores:** mais contexto, mas embeddings menos específicos.
- **Chunks menores:** mais precisão, mas risco de perder contexto.

### 8.2. Overlap vs. redundância

- **Mais overlap:** melhor preservação de contexto entre chunks.
- **Menos overlap:** menos redundância, menor uso de armazenamento.

### 8.3. Processamento vs. qualidade

- **Mais processamento:** chunks de melhor qualidade, mas maior custo computacional.
- **Menos processamento:** mais rápido, mas qualidade pode ser comprometida.

---

## 9. Conclusão

O pré-processamento é a fundação de um RAG eficaz. Chunks bem estruturados resultam em:

- **Melhor recuperação:** embeddings mais precisos encontram informações relevantes.
- **Contexto preservado:** chunks mantêm significado semântico coerente.
- **Metadados úteis:** rastreabilidade e debugging mais fáceis.
- **Performance otimizada:** tamanhos adequados para LLMs e bases vetoriais.

No próximo capítulo, vamos ver como construir prompts eficazes que combinam esses chunks recuperados com instruções claras para o LLM.

**Exercício prático:** modifique `rag_demo/ingest.py` para experimentar diferentes tamanhos de chunk (500, 1000, 1500) e compare a qualidade das respostas usando `scripts/run_query.py`.

---

### Pergunta ao leitor

Agora que dominamos o pré-processamento, quer que eu escreva o **Capítulo 05 — Prompting e Engenharia de Contexto** focando em como construir prompts eficazes para RAG?