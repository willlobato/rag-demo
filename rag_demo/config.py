#!/usr/bin/env python3
"""
⚙️ CONFIGURATION MANAGEMENT - Sistema de Configuração Educacional

Módulo centralizado de configuração que implementa boas práticas de gerenciamento
de configurações para sistemas RAG, demonstrando padrões profissionais de
configuração baseada em ambiente.

📚 FUNDAMENTAÇÃO CONCEITUAL:

Este módulo aplica o padrão "Configuration as Code" e "Environment-based Configuration"
para criar um sistema flexível e robusto de gerenciamento de configurações que
facilita desenvolvimento, testing e deployment em diferentes ambientes.

🎯 ARQUITETURA DE CONFIGURAÇÃO:

1️⃣ ENVIRONMENT-BASED CONFIGURATION:
   📝 CONCEITO: Configurações adaptáveis baseadas em variáveis de ambiente
   
   VANTAGENS:
   - Flexibilidade entre ambientes (dev/test/prod)
   - Segurança (credenciais não ficam no código)
   - Facilita CI/CD e containerização
   - Permite configuração externa sem rebuild
   
   IMPLEMENTAÇÃO:
   - os.getenv() com valores padrão
   - Type casting automático (int, float, bool)
   - Validação implícita de tipos
   - Fallbacks seguros para desenvolvimento

2️⃣ SEPARATION OF CONCERNS:
   📝 CONCEITO: Agrupamento lógico de configurações por responsabilidade
   
   CATEGORIAS IMPLEMENTADAS:
   - DIRETÓRIOS: Paths para dados e persistência
   - CHROMADB: Configurações do banco vetorial
   - MODELOS OLLAMA: Especificação de modelos de IA
   - CHUNKING: Parâmetros de fragmentação de texto
   - RETRIEVAL: Configurações de busca e recuperação
   - LLM: Parâmetros de geração de linguagem
   - OPERACIONAIS: Configurações de operação (reset, debug)

3️⃣ TWELVE-FACTOR APP COMPLIANCE:
   📝 CONCEITO: Seguimento dos princípios de aplicações cloud-native
   
   PRINCÍPIOS APLICADOS:
   - Config Factor: Configuração via ambiente
   - Backing Services: URLs e credenciais externalizadas
   - Port Binding: Configuração de portas via env vars
   - Disposability: Startup rápido com configs simples

📊 PADRÕES DE CONFIGURAÇÃO APLICADOS:

CONFIGURATION HIERARCHY:
```
1. Environment Variables (maior prioridade)
2. Default Values (fallback seguro)
3. Type Conversion (int, float, bool automático)
4. Validation (tipos e valores sensatos)
```

NAMING CONVENTION:
- SNAKE_CASE para variáveis Python
- SCREAMING_SNAKE_CASE para environment variables
- Prefixos lógicos por categoria
- Nomes descritivos e não ambíguos

DEFAULT VALUE STRATEGY:
- Valores sensatos para desenvolvimento local
- Configurações seguras (temperatura=0, não reset)
- Paths relativos para portabilidade
- Modelos padrão disponíveis localmente

🔧 CONFIGURAÇÕES DETALHADAS:

DIRETÓRIOS E PERSISTÊNCIA:
- DATA_DIR: Localização dos documentos fonte
- PERSIST_DIR: Diretório do banco vetorial ChromaDB
- Estratégia: Paths relativos para portabilidade

MODELOS DE IA:
- LLM_MODEL: Modelo de linguagem para geração
- EMB_MODEL: Modelo de embeddings para vetorização
- Padrão: llama3 + nomic-embed-text (disponíveis via Ollama)

PROCESSAMENTO DE TEXTO:
- CHUNK_SIZE: Tamanho dos fragmentos de texto
- CHUNK_OVERLAP: Sobreposição entre chunks
- Trade-off: Granularidade vs contexto

RECUPERAÇÃO DE INFORMAÇÃO:
- RETRIEVAL_K: Número de documentos recuperados
- COLLECTION_NAME: Nome da coleção no ChromaDB
- Balance: Recall vs precisão vs performance

GERAÇÃO DE LINGUAGEM:
- TEMPERATURE: Criatividade vs determinismo (0.0-1.0)
- Padrão: 0 para respostas determinísticas

OPERAÇÕES:
- RESET_CHROMA: Flag para reset do índice
- Segurança: Padrão false para evitar perda acidental

🧪 CASOS DE USO EDUCACIONAIS:

DESENVOLVIMENTO LOCAL:
```python
# Configuração automática para desenvolvimento
# Todos os valores padrão funcionam out-of-the-box
```

EXPERIMENTAÇÃO:
```bash
# Teste com diferentes parâmetros
export CHUNK_SIZE=1000
export RETRIEVAL_K=10
export TEMPERATURE=0.3
```

AMBIENTE DE PRODUÇÃO:
```bash
# Configuração otimizada para produção
export DATA_DIR=/app/data
export PERSIST_DIR=/app/db
export LLM_MODEL=llama3:70b
```

TESTING AUTOMATIZADO:
```bash
# Configuração para testes
export RESET_CHROMA=1
export COLLECTION_NAME=test-collection
```

📈 MONITORAMENTO E DEBUGGING:

CONFIGURAÇÃO VISÍVEL:
- Todas as configs são facilmente auditáveis
- Valores ativos podem ser inspecionados em runtime
- Environment variables documentadas in-line

VALIDAÇÃO IMPLÍCITA:
- Type conversion automática previne erros
- Valores padrão sempre válidos
- Fallbacks seguros para todos os parâmetros

FLEXIBILIDADE OPERACIONAL:
- Mudanças sem restart (para algumas configs)
- Override seletivo de parâmetros
- Configuração por feature flags

🚀 VALOR EDUCACIONAL:

Este módulo demonstra:
1. Padrões profissionais de gerenciamento de configuração
2. Aplicação dos princípios Twelve-Factor App
3. Estratégias de configuração environment-based
4. Type safety e validation em Python
5. Organização modular de configurações

O design facilita experimentação educacional, deployment profissional
e manutenção de longo prazo, servindo como template para sistemas
de configuração robustos.
"""

import os
from pathlib import Path

# Diretórios e Persistência
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
PERSIST_DIR = os.getenv("PERSIST_DIR", "db")

# ChromaDB Configuration
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "demo-rag")

# Modelos Ollama
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
EMB_MODEL = os.getenv("EMB_MODEL", "nomic-embed-text")

# Configurações de Chunking (Fragmentação de Texto)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# Configurações de Retrieval (Recuperação)
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))

# Configurações do LLM (Large Language Model)
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

# Configurações Operacionais
RESET_CHROMA = os.getenv("RESET_CHROMA", "0") == "1"

# Ollama (opcional)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
