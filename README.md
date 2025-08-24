# RAG Demo - Sistema Educacional Completo

> **Sistema RAG (Retrieval-Augmented Generation) com documentação educacional abrangente**

Um projeto de demonstração completo de RAG usando Ollama para modelos locais, ChromaDB para armazenamento vetorial e LangChain para orquestração, acompanhado de um sistema educacional estruturado para aprendizagem progressiva.

---

## 📚 **DOCUMENTAÇÃO COMPLETA**

### 🎯 **Portal Principal**: **[docs/](docs/)** 
> Acesse toda a documentação organizada por categorias

### 🚀 **Links Rápidos**
- **[docs/guides/](docs/guides/)** 📖 - Tutoriais e guias práticos
- **[docs/reference/](docs/reference/)** 📋 - Documentação técnica detalhada  
- **[docs/organization/](docs/organization/)** 🗄️ - Organização e navegação do sistema

---

## 🚀 **INÍCIO RÁPIDO**

### 📖 **Para Iniciantes**
```
1️⃣ docs/guides/TUTORIAL_RAG.md → Fundamentos
2️⃣ docs/reference/GLOSSARIO_CONCEITOS.md → Vocabulário
3️⃣ docs/guides/EXEMPLOS_USO_SCRIPTS.md → Prática
```

### 🔬 **Para Pesquisadores**  
```
1️⃣ docs/reference/DOCUMENTACAO_SCRIPTS_AVANCADOS.md → Análise técnica
2️⃣ docs/organization/CLASSIFICACAO_SCRIPTS.md → Complexidade
3️⃣ scripts/experiment.py → Experimentação
```

---

## 📋 **Sobre o Projeto**

Este projeto implementa um sistema RAG completo que:

- **Processa documentos** (TXT, MD, PDF) e os divide em chunks
- **Gera embeddings** usando o modelo `nomic-embed-text` do Ollama
- **Armazena vetores** no ChromaDB com persistência automática
- **Responde perguntas** usando o modelo `llama3` com contexto recuperado
- **Fornece ferramentas educacionais** para aprendizagem estruturada de RAG

## 🏗️ **Arquitetura do Sistema**

```
📦 RAG Demo System
├── 🏗️ rag_demo/          # Core package (módulos principais)
├── 📂 scripts/           # Scripts de automação (12 total)
├── 📚 docs/              # Documentação educacional completa
├── 📊 data/              # Documentos de exemplo
└── 🗄️ db/               # Base de dados vetorial (ChromaDB)
```

### 📦 **Core Package (rag_demo/)**
- **`config.py`** - Configuração baseada em ambiente
- **`ingest.py`** - Pipeline de ingestão de documentos
- **`rag.py`** - Sistema RAG completo (retrieval + generation)
- **`utils.py`** - Funções utilitárias

### 📂 **Scripts Organizados por Complexidade**
- **🟢 Básico (4):** Operações fundamentais
- **🟡 Intermediário (3):** Análise e métricas
- **🔴 Avançado (5):** Pesquisa e experimentação

---

## 🛠️ **Instalação e Setup**

### 1. **Ollama**
```bash
# Instalar Ollama (macOS)
brew install ollama

# Iniciar o serviço
brew services start ollama

# Baixar os modelos
ollama pull llama3
ollama pull nomic-embed-text
```

### 2. **Python Environment**
```bash
# Clonar o repositório
git clone <repo-url>
cd rag-demo

# Criar ambiente virtual (recomendado: .venv)
python -m venv .venv

# Ativar o ambiente
# macOS / Linux (bash, zsh)
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (cmd)
.venv\Scripts\activate.bat

# Instalar dependências
pip install -r requirements.txt
```

### VSCode (opcional)

- Abra a Command Palette e selecione "Python: Select Interpreter"; escolha o interpretador dentro de `.venv`.
- Para ativação automática do ambiente no terminal integrado, adicione em seu `settings.json` (ou ative a opção nas configurações):

```json
{
	"python.terminal.activateEnvironment": true
}
```

O plugin Python do VSCode normalmente ativa o ambiente automaticamente quando o interpretador correto é selecionado.

### 3. **Verificar Instalação**
```bash
# Verificar se Ollama está rodando
curl http://localhost:11434/api/tags

# Testar o sistema RAG
python -m rag_demo.rag "O que é RAG?"
```

---

## 🚀 **Uso Básico**

### **Ingestão de Documentos**
```bash
# Adicionar documentos à pasta data/
python -m rag_demo.ingest

# Ou usar script dedicado
python scripts/run_ingest.py
```

### **Consultas RAG**
```bash
# Consulta simples
python -m rag_demo.rag "Explique embeddings vetoriais"

# Usando script com opções avançadas
python scripts/run_query.py "Como funciona a recuperação vetorial?"
```

### **Análise e Métricas**
```bash
# Listar documentos
python scripts/list_docs.py

# Analisar chunks
python scripts/analyze_chunks.py

# Avaliar performance
python scripts/evaluate_rag.py
```

---

## 📊 **Sistema Educacional**

### 🎓 **Níveis de Aprendizagem**
- **🟢 Básico:** Conceitos fundamentais e operações simples
- **🟡 Intermediário:** Análise, métricas e otimização
- **🔴 Avançado:** Pesquisa, experimentação e desenvolvimento

### 📖 **Material Disponível**
- **37 arquivos** de documentação total
- **Tutoriais interativos** com exemplos práticos
- **Glossário técnico** com 50+ conceitos
- **Guias de navegação** para diferentes perfis

### 🧭 **Como Navegar**
1. **Iniciante?** Comece em `docs/guides/TUTORIAL_RAG.md`
2. **Desenvolvedor?** Vá para `docs/reference/DOCUMENTACAO_SCRIPTS_AVANCADOS.md`
3. **Pesquisador?** Explore `docs/reference/DOCUMENTACAO_SCRIPTS_AVANCADOS.md`

---

## 🔧 **Desenvolvimento**

### **Estrutura de Código**
```python
from rag_demo import config, ingest, rag, utils

# Configuração
cfg = config.get_config()

# Ingestão
docs = ingest.load_and_process_documents()

# Consulta RAG
response = rag.query("Sua pergunta aqui")
```

### **Extensibilidade**
- Adicione novos loaders em `utils.py`
- Configure novos modelos em `config.py`
- Desenvolva métricas em `scripts/`

---

## 📈 **Performance e Métricas**

### **Benchmarks Incluídos**
- **Similarity Analysis:** Análise de similaridade entre chunks
- **Retrieval Evaluation:** Avaliação da qualidade de recuperação
- **Generation Metrics:** Métricas de qualidade de geração

### **Ferramentas de Análise**
- `scripts/analyze_similarity.py` - Matriz de similaridade
- `scripts/evaluate_rag.py` - Métricas completas
- `scripts/experiment.py` - Framework experimental

---

## 📝 **Licença**

Este projeto é open source e está disponível sob a licença MIT.

---

## 🤝 **Contribuição**

1. **Fork** o projeto
2. **Crie** uma branch para sua feature
3. **Documente** seguindo os padrões educacionais
4. **Submeta** um pull request

Para contribuições, consulte a documentação em `docs/` para diretrizes do projeto.

---

**💻 Desenvolvido com foco educacional para aprendizagem progressiva de sistemas RAG**
