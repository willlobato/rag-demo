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

## ⚡ **TESTE RÁPIDO (2 minutos)**

```bash
# 1. Indexar dados de exemplo
python scripts/run_ingest.py

# 2. Fazer primeira pergunta
python scripts/run_query.py "Qual é a latência média das APIs?"

# 3. Ver como dados foram divididos
python scripts/analyze_chunks.py

# 4. Buscar por tecnologia específica (busca os 3 chunks mais similares)
python scripts/search_docs.py "Kubernetes" 3
```

---

## 🚀 **INÍCIO RÁPIDO**

### 📖 **Para Iniciantes (com exemplo prático)**
```
1️⃣ docs/guides/TUTORIAL_RAG.md → Fundamentos de RAG
2️⃣ python scripts/run_ingest.py → Indexar dados de exemplo
3️⃣ python scripts/run_query.py "Como funciona o cache distribuído?" → Primeira consulta
4️⃣ docs/guides/EXEMPLOS_USO_SCRIPTS.md → Explorar mais funcionalidades
```

### 🔬 **Para Desenvolvedores (análise técnica)**  
```
1️⃣ python scripts/analyze_chunks.py --full → Ver como dados foram processados
2️⃣ python scripts/show_vectors.py "microserviços" true → Entender embeddings
3️⃣ docs/reference/DOCUMENTACAO_SCRIPTS_AVANCADOS.md → Análise avançada
4️⃣ scripts/experiment.py → Experimentação com parâmetros
```

---

## 📋 **Sobre o Projeto**

Este projeto implementa um sistema RAG completo que permite **consultar documentos de forma inteligente**. Usando o texto de exemplo incluído (`data/sistema_completo.txt`), você pode fazer perguntas e receber respostas contextualizadas.

**O sistema:**
- **Processa documentos** (TXT, MD, PDF) e os divide em chunks otimizados
- **Gera embeddings** usando o modelo `nomic-embed-text` do Ollama
- **Armazena vetores** no ChromaDB com persistência automática
- **Responde perguntas** usando o modelo `llama3` com contexto recuperado
- **Fornece ferramentas educacionais** para aprendizagem estruturada de RAG

**Exemplo de uso com nosso texto de demonstração:**
- 📊 "Qual é a latência média das APIs do sistema?"
- 🏗️ "Como foi implementada a arquitetura de microserviços?"
- 🔒 "Quais tecnologias foram usadas para segurança?"
- 📈 "Quantos usuários simultâneos o sistema suporta?"

## 🏗️ **Arquitetura do Sistema**

```
📦 RAG Demo System
├── 🏗️ rag_demo/          # Core package (módulos principais)
├── 📂 scripts/           # Scripts de automação (12 total)
├── 📚 docs/              # Documentação educacional completa
├── 📊 data/              # Documentos de exemplo
│   └── sistema_completo.txt    # 📄 Texto de exemplo para demonstração
└── 🗄️ db/               # Base de dados vetorial (ChromaDB)
```

### 📄 **Texto de Exemplo Incluído**

O arquivo `data/sistema_completo.txt` contém um texto de demonstração simples sobre um sistema fictício com:

- **🚀 Performance:** Exemplo de métricas (10k usuários, 150ms latência)  
- **🏗️ Arquitetura:** Conceitos básicos (Spring Boot, Docker, Kubernetes)
- **🗄️ Dados:** Tecnologias comuns (PostgreSQL, Redis cache)
- **📊 Monitoramento:** Stack típico (Prometheus, Grafana, Slack)
- **🔒 Segurança:** Padrões básicos (JWT, OAuth2, rate limiting)
- **⚙️ DevOps:** Ferramentas usuais (Jenkins CI/CD, deployment)

**Perguntas de exemplo que você pode testar:**
- "Como foi otimizado o processo de login?"
- "Qual a arquitetura de microserviços utilizada?" 
- "Quais são as métricas de performance do sistema?"
- "Como funciona o sistema de monitoramento?"

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

# Testar o sistema RAG com dados reais
python -m rag_demo.rag "Qual tecnologia é usada para cache distribuído?"

# Resultado esperado: resposta sobre Infinispan e Redis
```

---

## 🚀 **Uso Básico**

### **1. Ingestão dos Documentos**
```bash
# Indexar o documento de exemplo (sistema_completo.txt já está em data/)
python scripts/run_ingest.py

# Verificar o que foi indexado
python scripts/list_docs.py
```

### **2. Consultas RAG - Exemplos Práticos**
```bash
# Perguntas sobre performance do sistema
python scripts/run_query.py "Qual é a latência média das APIs?"
python scripts/run_query.py "Quantos usuários simultâneos o sistema suporta?"

# Perguntas sobre arquitetura  
python scripts/run_query.py "Como foi implementada a arquitetura de microserviços?"
python scripts/run_query.py "Quais tecnologias são usadas para cache?"

# Perguntas sobre monitoramento
python scripts/run_query.py "Como funciona o sistema de monitoramento?"
python scripts/run_query.py "Onde são enviados os alertas críticos?"

# Perguntas sobre segurança
python scripts/run_query.py "Quais mecanismos de segurança foram implementados?"
python scripts/run_query.py "Como é feita a autenticação dos usuários?"
```

### **3. Análise dos Dados Indexados**
```bash
# Ver como o documento foi dividido em chunks
python scripts/analyze_chunks.py --full

# Buscar tópicos específicos sem usar LLM
python scripts/search_docs.py "PostgreSQL" 3      # Buscar "PostgreSQL" e mostrar os 3 resultados mais similares
python scripts/search_docs.py "Kubernetes" 5      # Buscar "Kubernetes" e mostrar os 5 resultados mais similares

# Sintaxe: python scripts/search_docs.py "TERMO_DE_BUSCA" NÚMERO_DE_RESULTADOS
# O número é opcional (padrão: 5) e determina quantos chunks mais similares mostrar

# Analisar embeddings e vetores
python scripts/show_vectors.py "microserviços" true

# 🛡️ RAG com proteções contra alucinações
python scripts/rag_with_guardrails.py "Como funciona o cache distribuído?" strict

# 📊 Otimizar threshold automaticamente para seus dados
python scripts/threshold_optimizer.py

# 📖 Ver guia completo de uso de guardrails
# docs/guides/USO_GUARDRAILS.md - Exemplos práticos testados
# docs/reference/THRESHOLD_OPTIMIZER_GUIDE.md - Guia completo do threshold optimizer
# docs/reference/RELATORIO_TESTES_GUARDRAILS.md - Validação com dados reais
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

# Consulta RAG com texto de exemplo
response = rag.query("Como foi otimizado o processo de login?")
print(response)  # Resposta baseada no conteúdo do texto de exemplo
```

### **Experimentação com Diferentes Configurações**
```bash
# Testar chunks menores para mais precisão
export CHUNK_SIZE=300
export CHUNK_OVERLAP=60
python scripts/run_ingest.py
python scripts/run_query.py "Qual é o uptime do serviço?"

# Testar recuperação de mais chunks
export RETRIEVAL_K=6
python scripts/run_query.py "Explique toda a stack de tecnologias usadas"

# Analisar impacto das mudanças
python scripts/analyze_chunks.py --full
```

### **Extensibilidade**
- Adicione seus próprios documentos em `data/`
- Configure novos modelos em `config.py`
- Desenvolva métricas específicas em `scripts/`
- Experimente com diferentes tipos de conteúdo (APIs, manuais, documentação técnica real, etc.)

---

## 📈 **Performance e Métricas**

### **Análise do Texto de Exemplo**
Com o texto `sistema_completo.txt`, você pode testar:

```bash
# Análise de similaridade entre chunks sobre diferentes tecnologias
python scripts/analyze_similarity.py

# Avaliação da qualidade das respostas
python scripts/evaluate_rag.py

# Experimentação com diferentes parâmetros
python scripts/experiment.py
```

### **Testes Práticos de Demonstração**
- **Similarity Analysis:** Compare chunks sobre "Spring Boot" vs "PostgreSQL"
- **Retrieval Quality:** Teste perguntas sobre performance vs arquitetura
- **Response Quality:** Avalie respostas sobre tópicos específicos (ex: segurança)

### **Ferramentas de Análise**
```bash
# Matriz de similaridade entre conceitos do texto
python scripts/analyze_similarity.py

# Métricas de recuperação para diferentes tipos de pergunta
python scripts/analyze_retrieval.py  

# Experimentos com variações de consulta
python scripts/experiment.py --topic "performance"
```

### **Casos de Teste com o Texto de Exemplo**
- 🔍 **Busca específica:** "Prometheus" vs "monitoramento"
- 🏗️ **Arquitetura:** Perguntas sobre microserviços e containers
- 📊 **Métricas:** Consultas sobre performance e uptime
- 🔒 **Segurança:** Questões sobre JWT, OAuth2, rate limiting

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
