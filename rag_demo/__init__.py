#!/usr/bin/env python3
"""
🎯 RAG DEMO CORE PACKAGE - Arquitetura Modular Educacional

Pacote principal que implementa a arquitetura core do sistema RAG, fornecendo
uma interface limpa e modular para todas as funcionalidades de recuperação 
e geração aumentada.

📚 FUNDAMENTAÇÃO ARQUITETURAL:

Este módulo implementa o padrão de arquitetura modular para sistemas RAG,
separando responsabilidades em módulos especializados e fornecendo uma
API unificada para interação com todas as funcionalidades do sistema.

🏗️ ARQUITETURA DO PACKAGE:

O package rag_demo segue princípios de design modular bem estabelecidos:

1️⃣ SEPARAÇÃO DE RESPONSABILIDADES:
   📝 CONCEITO: Cada módulo tem uma responsabilidade específica e bem definida
   
   ESTRUTURA MODULAR:
   - config.py: Configurações centralizadas e gerenciamento de ambiente
   - ingest.py: Pipeline de ingestão e processamento de documentos
   - rag.py: Engine de recuperação e geração de respostas
   - utils.py: Utilitários compartilhados e funções auxiliares
   - __init__.py: Interface pública e controle de exportações

2️⃣ INTERFACE PÚBLICA LIMPA:
   📝 CONCEITO: API simplificada que esconde complexidade interna
   
   FUNÇÕES EXPORTADAS:
   - ingest_documents(): Pipeline completo de ingestão
   - query_rag(): Interface principal de consulta
   - search_similar_docs(): Busca semântica avançada
   - list_documents(): Exploração do índice criado
   
   🎯 VANTAGENS:
   - Abstração da complexidade interna
   - Interface consistente para todos os usos
   - Facilita manutenção e evolução
   - Permite reutilização em diferentes contextos

3️⃣ GERENCIAMENTO DE DEPENDÊNCIAS:
   📝 CONCEITO: Controle explícito de importações e exportações
   
   ESTRATÉGIA DE IMPORTAÇÃO:
   - Importações explícitas de módulos internos
   - Configurações carregadas automaticamente
   - Funções principais disponibilizadas diretamente
   - Controle de namespace via __all__

📊 PADRÕES DE DESIGN APLICADOS:

FACADE PATTERN:
- Simplifica interface para subsistema complexo
- Unifica acesso a múltiplos módulos especializados
- Reduz acoplamento entre cliente e implementação

MODULE PATTERN:
- Encapsula funcionalidades relacionadas
- Controla visibilidade de componentes internos
- Facilita testing e mocking

DEPENDENCY INJECTION:
- Configurações injetadas via config.py
- Facilita testing com configurações alternativas
- Permite customização sem modificar código

🎯 CASOS DE USO EDUCACIONAIS:

APRENDIZAGEM DE ARQUITETURA:
- Demonstra padrões de design em Python
- Mostra organização de packages Python
- Exemplifica separação de responsabilidades

DESENVOLVIMENTO MODULAR:
- Template para sistemas similares
- Base para extensões e customizações
- Referência para boas práticas

INTEGRAÇÃO SISTEMÁTICA:
- Interface única para múltiplas funcionalidades
- Ponto de entrada consistente para scripts
- Base para APIs e interfaces externas

🚀 VALOR EDUCACIONAL:

Este arquivo demonstra:
1. Como estruturar packages Python profissionalmente
2. Aplicação de padrões de design em sistemas reais
3. Criação de APIs limpas e reutilizáveis
4. Separação efetiva de responsabilidades
5. Gerenciamento de dependências em Python

O design modular facilita aprendizagem, manutenção e extensão do sistema,
servindo como exemplo de arquitetura bem estruturada para sistemas RAG.
"""

from .config import *
from .ingest import ingest_documents
from .rag import query_rag, search_similar_docs, list_documents

__version__ = "0.1.0"
__author__ = "Will Lobato"
__description__ = "Sistema RAG Educacional com Ollama e ChromaDB"

# Exportar funções principais
__all__ = [
    "ingest_documents",
    "query_rag", 
    "search_similar_docs",
    "list_documents"
]
