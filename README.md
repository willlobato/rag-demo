# RAG Demo - Sistema Educacional Completo

### 📖 **Documentação Técnica Complementar**

### 🎯 **Portal Principal**: **[docs/](docs/)** 
> Acesso completo à documentação organizada por categorias

### 🚀 **Links Rápidos da Documentação**

#### 📖 **[docs/guides/](docs/guides/) - Guias e Tutoriais Práticos**
- **[TUTORIAL_RAG.md](docs/guides/TUTORIAL_RAG.md)** 📚 - Tutorial completo sobre conceitos RAGTécnica Complementar**

### 🎯 **Portal Principal**: **[docs/](docs/)** 
> Acesso completo à documentação organizada por categorias

### 🚀 **Links Rápidos da Documentação**

#### 📖 **[docs/guides/](docs/guides/) - Guias e Tutoriais Práticos**
- **[TUTORIAL_RAG.md](docs/guides/TUTORIAL_RAG.md)** 📚 - Tutorial completo sobre conceitos RAG
- **[GUIA_NAVEGACAO.md](docs/guides/GUIA_NAVEGACAO.md)** 🧭 - Fluxos de navegação e aprendizagem
- **[EXEMPLOS_USO_SCRIPTS.md](docs/guides/EXEMPLOS_USO_SCRIPTS.md)** 💡 - Exemplos práticos de uso dos scripts
- **[USO_GUARDRAILS.md](docs/guides/USO_GUARDRAILS.md)** 🛡️ - Guia de uso dos guardrails anti-alucinaçãom documentação educacional abrangente**

Um projeto de demonstração completo de RAG usando Ollama para modelos locais, ChromaDB para armazenamento vetorial e LangChain para orquestração, acompanhado de um sistema educacional estruturado O material em **[docs/learning/llm-to-rag/](docs/learning/llm-to-rag/)** oferece uma jornada estruturada do básico ao avançado:

- **📚 Teoria + Prática:** Cada conceito acompanha exemplos executáveis
- **🔄 Progressivo:** Conhecimento construído incrementalmente
- **🛠️ Hands-on:** Scripts práticos para cada etapa
- **🎯 Focado:** Do LLM básico ao RAG em produçãoprendizagem progressiva.

---

## 🚀 **Por que RAG? Por que agora?**

Imagine fazer estas perguntas a uma IA e receber respostas **precisas** e **com fontes**:

- 🏥 **"Qual o protocolo específico para pacientes com COVID-19 internados na UTI segundo as diretrizes atualizadas de janeiro de 2024?"**
- 💼 **"Com base nos contratos da empresa, qual a cláusula exata sobre rescisão em caso de descumprimento do prazo de entrega?"**
- 🔬 **"Considerando todos os papers sobre CRISPR publicados nos últimos 6 meses, quais são as 3 principais limitações técnicas ainda não resolvidas?"**
- 📊 **"Analisando nossos dados internos de vendas, qual estratégia de precificação teve melhor resultado no Q3 de 2024?"**

**O problema:** Modelos de IA tradicionais (GPT, Claude, etc.) não conseguem responder essas perguntas porque:
- ❌ **Não têm acesso aos seus dados privados** (contratos, relatórios internos, bases de conhecimento específicas)
- ❌ **Ficam desatualizados rapidamente** (dados de treinamento têm uma "data de corte")
- ❌ **Podem "alucinar" informações** quando não sabem a resposta
- ❌ **Não fornecem fontes verificáveis** para auditoria

**A solução RAG:**
- ✅ **Conecta IA aos seus dados em tempo real** - documentos, PDFs, bases de conhecimento, APIs
- ✅ **Sempre atualizado** - busca as informações mais recentes no momento da pergunta
- ✅ **Respostas com fontes** - você sabe exatamente de onde veio cada informação
- ✅ **Reduz alucinações drasticamente** - a IA responde baseada em evidências concretas

### 🎯 **Casos Reais Que Só RAG Resolve**

**🏢 Consultoria Jurídica Inteligente**
- Cliente pergunta sobre uma lei específica → RAG busca na base atualizada de legislação → resposta precisa com artigos e incisos exatos
- **Sem RAG:** IA pode inventar leis que não existem ou citar versões antigas

**🏥 Assistente Médico com Protocolos Atualizados**
- Médico consulta sobre novo tratamento → RAG acessa diretrizes médicas mais recentes → recomendação baseada em evidência científica atual
- **Sem RAG:** Informação médica desatualizada pode ser perigosa

**📈 Business Intelligence Conversacional**
- "Qual produto teve melhor margem no último trimestre?" → RAG consulta dados de vendas reais → análise baseada em números concretos
- **Sem RAG:** IA não tem acesso aos dados internos da empresa

**🎓 Assistente Educacional Personalizado**
- Estudante pergunta sobre tópico específico → RAG busca em bibliografia do curso + anotações do professor → resposta contextualizada ao currículo
- **Sem RAG:** Respostas genéricas que não seguem a metodologia do curso

---

## 📚 **SISTEMA EDUCACIONAL COMPLETO**

### 🎓 **Trilha de Aprendizagem: Do LLM ao RAG**

**NOVO!** Material educacional estruturado em **[docs/learning/llm-to-rag/](docs/learning/llm-to-rag/)** com 13 capítulos progressivos:

#### 🚀 **Trilha Básica (Para Iniciantes)**
1. **[00-introducao.md](docs/learning/llm-to-rag/00-introducao.md)** - Por que RAG? Visão geral e primeiros passos
2. **[01-llms-basico.md](docs/learning/llm-to-rag/01-llms-basico.md)** - LLMs, tokens, contexto e LangChain
3. **[02-embeddings-similaridade.md](docs/learning/llm-to-rag/02-embeddings-similaridade.md)** - Como transformar texto em vetores
4. **[03-bases-vetoriais.md](docs/learning/llm-to-rag/03-bases-vetoriais.md)** - ChromaDB e busca por similaridade

#### 🔧 **Trilha Técnica (Para Implementadores)**
5. **[04-nlp-preprocessamento.md](docs/learning/llm-to-rag/04-nlp-preprocessamento.md)** - Chunking, limpeza e normalização
6. **[05-rag-basico.md](docs/learning/llm-to-rag/05-rag-basico.md)** - Pipeline RAG completo com diagrama
7. **[06-rag-avancado.md](docs/learning/llm-to-rag/06-rag-avancado.md)** - Estratégias avançadas e otimizações
8. **[07-guardrails.md](docs/learning/llm-to-rag/07-guardrails.md)** - Evitando alucinações

#### 🏭 **Trilha Avançada (Para Produção)**
9. **[08-deploy.md](docs/learning/llm-to-rag/08-deploy.md)** - Deploy e monitoramento
10. **[09-prompt-context-engineering.md](docs/learning/llm-to-rag/09-prompt-context-engineering.md)** - Engenharia de prompts
11. **[10-estudo-de-caso.md](docs/learning/llm-to-rag/10-estudo-de-caso.md)** - Caso prático completo
12. **[11-avaliacao-rag.md](docs/learning/llm-to-rag/11-avaliacao-rag.md)** - Métricas e avaliação (RAGAs)
13. **[12-troubleshooting.md](docs/learning/llm-to-rag/12-troubleshooting.md)** - Problemas comuns e soluções
14. **[prompt_cheatsheet.md](docs/learning/llm-to-rag/prompt_cheatsheet.md)** - Referência rápida

### 📖 **Documentação Técnica Complementar**

### 📖 **Documentação Técnica Complementar**

### 🎯 **Portal Principal**: **[docs/](docs/)** 
> Acesso completo à documentação organizada por categorias

### 🚀 **Links Rápidos da Documentação**

#### 📖 **[docs/guides/](docs/guides/) - Guias e Tutoriais Práticos**
- **[TUTORIAL_RAG.md](docs/guides/TUTORIAL_RAG.md)** � - Tutorial completo sobre conceitos RAG
- **[GUIA_NAVEGACAO.md](docs/guides/GUIA_NAVEGACAO.md)** 🧭 - Fluxos de navegação e aprendizagem
- **[EXEMPLOS_USO_SCRIPTS.md](docs/guides/EXEMPLOS_USO_SCRIPTS.md)** 💡 - Exemplos práticos de uso dos scripts
- **[USO_GUARDRAILS.md](docs/guides/USO_GUARDRAILS.md)** 🛡️ - Guia de uso dos guardrails anti-alucinação

#### 📋 **[docs/reference/](docs/reference/) - Documentação de Referência**
- **[GLOSSARIO_CONCEITOS.md](docs/reference/GLOSSARIO_CONCEITOS.md)** � - Dicionário de termos técnicos
- **[DOCUMENTACAO_SCRIPTS_AVANCADOS.md](docs/reference/DOCUMENTACAO_SCRIPTS_AVANCADOS.md)** 🔬 - Análise técnica avançada
- **[GUARDRAILS_GUIDE.md](docs/reference/GUARDRAILS_GUIDE.md)** 🛡️ - Guia técnico detalhado de guardrails
- **[THRESHOLD_OPTIMIZER_GUIDE.md](docs/reference/THRESHOLD_OPTIMIZER_GUIDE.md)** 🎯 - Guia do otimizador de threshold
- **[RELATORIO_TESTES_GUARDRAILS.md](docs/reference/RELATORIO_TESTES_GUARDRAILS.md)** 📊 - Relatório de validação dos guardrails

#### 🗄️ **[docs/organization/](docs/organization/) - Meta-Documentação**
- **[INDICE_DOCUMENTACAO.md](docs/organization/INDICE_DOCUMENTACAO.md)** � - Índice geral completo
- **[CLASSIFICACAO_SCRIPTS.md](docs/organization/CLASSIFICACAO_SCRIPTS.md)** 🏷️ - Classificação por complexidade
- **[MAPA_VISUAL.md](docs/organization/MAPA_VISUAL.md)** 🗺️ - Representação visual do sistema
- **[RELATORIO_ORGANIZACAO.md](docs/organization/RELATORIO_ORGANIZACAO.md)** 📈 - Relatório organizacional
- **[RESUMO_SISTEMA_EDUCACIONAL.md](docs/organization/RESUMO_SISTEMA_EDUCACIONAL.md)** 🎓 - Visão geral educacional

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

### 📖 **Para Iniciantes (com trilha educacional)**
```
1️⃣ docs/learning/llm-to-rag/00-introducao.md → Fundamentos teóricos
2️⃣ docs/learning/llm-to-rag/01-llms-basico.md → Entender LLMs e LangChain
3️⃣ python scripts/run_ingest.py → Primeira prática
4️⃣ python scripts/run_query.py "Como funciona o cache distribuído?" → Ver RAG funcionando
5️⃣ docs/learning/llm-to-rag/05-rag-basico.md → Compreender o pipeline completo
```

### 🔬 **Para Desenvolvedores (análise técnica)**  
```
1️⃣ docs/learning/llm-to-rag/02-embeddings-similaridade.md → Entender embeddings
2️⃣ python scripts/analyze_chunks.py --full → Ver processamento na prática
3️⃣ python scripts/show_vectors.py "microserviços" true → Entender vetorização
4️⃣ docs/learning/llm-to-rag/06-rag-avancado.md → Técnicas avançadas
5️⃣ docs/reference/DOCUMENTACAO_SCRIPTS_AVANCADOS.md → Scripts para produção
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

### 🎓 **Trilha Educacional Completa (13 Capítulos)**
O material em **[docs/learning/llm-to-rag/](docs/learning/llm-to-rag/)** oferece uma jornada estruturada do básico ao avançado:

- **� Teoria + Prática:** Cada conceito acompanha exemplos executáveis
- **� Progressivo:** Conhecimento construído incrementalmente
- **�️ Hands-on:** Scripts práticos para cada etapa
- **🎯 Focado:** Do LLM básico ao RAG em produção

### 📖 **Material Disponível Total**
- **50+ arquivos** de documentação
- **Tutoriais interativos** com exemplos práticos
- **Glossário técnico** com conceitos essenciais
- **Guias de navegação** para diferentes perfis
- **Casos de estudo** com problemas reais

### 🧭 **Como Navegar o Aprendizado**
1. **🟢 Iniciante?** Comece em `docs/learning/llm-to-rag/00-introducao.md`
2. **🟡 Desenvolvedor?** Vá para `docs/learning/llm-to-rag/05-rag-basico.md`
3. **🔴 Avançado?** Explore `docs/learning/llm-to-rag/11-avaliacao-rag.md`

### 🏆 **Níveis de Conhecimento**
- **🟢 Básico:** Conceitos fundamentais e operações simples
- **🟡 Intermediário:** Análise, métricas e otimização  
- **🔴 Avançado:** Pesquisa, experimentação e produção

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
