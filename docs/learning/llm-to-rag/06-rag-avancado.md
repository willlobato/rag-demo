# Capítulo 06 — RAG Avançado: Técnicas e Otimizações para Produção

No capítulo anterior, construímos um sistema RAG básico funcional. Agora vamos elevá-lo ao próximo nível, implementando técnicas avançadas que fazem a diferença entre um protótipo e um sistema robusto de produção.

Este capítulo explora **guardrails**, **multi-retrieval**, **reranking**, **hibridização de busca** e outras técnicas que transformam um RAG simples em uma solução empresarial confiável.

**Resumo:** neste capítulo implementamos técnicas avançadas de RAG, incluindo guardrails, busca híbrida, reranking, validação de qualidade e otimizações para sistemas de produção.

**Sumário:**
1. Guardrails: controle de qualidade automatizado
2. Busca híbrida (vetorial + lexical)
3. Reranking inteligente
4. Multi-retrieval e fusão de resultados
5. Validação e filtragem de respostas
6. Memória conversacional
7. Métricas e monitoramento avançado
8. Otimizações para produção
9. Conclusão

---

## 1. Guardrails: controle de qualidade automatizado

Guardrails são validações automáticas que garantem que o sistema não produza respostas inadequadas, perigosas ou fora do escopo.

### 1.1. Validação de entrada

```python
import re
from typing import Tuple, List

class InputValidator:
    def __init__(self):
        self.blocked_patterns = [
            r'\b(hack|crack|bypass|exploit)\b',  # Termos de segurança
            r'\b(password|senha|token)\b',       # Informações sensíveis
            r'<script.*?>.*?</script>',          # XSS attempts
        ]
        
        self.min_length = 3
        self.max_length = 1000
    
    def validate_query(self, query: str) -> Tuple[bool, str]:
        """Valida se a consulta é segura e apropriada"""
        
        # Verifica tamanho
        if len(query) < self.min_length:
            return False, "Consulta muito curta"
        
        if len(query) > self.max_length:
            return False, "Consulta muito longa"
        
        # Verifica padrões bloqueados
        for pattern in self.blocked_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, f"Consulta contém termo não permitido"
        
        # Verifica se tem conteúdo textual
        if not any(c.isalpha() for c in query):
            return False, "Consulta deve conter texto"
        
        return True, "OK"
    
    def sanitize_query(self, query: str) -> str:
        """Limpa e normaliza a consulta"""
        # Remove caracteres especiais perigosos
        query = re.sub(r'[<>"\';]', '', query)
        
        # Normaliza espaços
        query = ' '.join(query.split())
        
        return query.strip()
```

### 1.2. Validação de contexto recuperado

```python
class ContextValidator:
    def __init__(self):
        self.min_relevance_score = 0.3
        self.max_chunks = 5
        
    def validate_context(self, chunks: List, query: str) -> List:
        """Valida e filtra chunks recuperados"""
        if not chunks:
            return chunks
        
        validated_chunks = []
        
        for chunk in chunks[:self.max_chunks]:
            # Calcula relevância baseada em sobreposição de palavras
            relevance = self._calculate_relevance(chunk.page_content, query)
            
            if relevance >= self.min_relevance_score:
                chunk.metadata['relevance_score'] = relevance
                validated_chunks.append(chunk)
        
        return validated_chunks
    
    def _calculate_relevance(self, content: str, query: str) -> float:
        """Calcula score de relevância simples"""
        content_words = set(content.lower().split())
        query_words = set(query.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(content_words.intersection(query_words))
        return overlap / len(query_words)
```

### 1.3. Validação de resposta

```python
class ResponseValidator:
    def __init__(self):
        self.blocked_responses = [
            "não posso ajudar com isso",
            "isso está fora do meu escopo",
            "não tenho informações suficientes"
        ]
        
        self.required_evidence_threshold = 0.2
    
    def validate_response(self, response: str, context_chunks: List) -> Tuple[bool, str, float]:
        """Valida se a resposta é adequada e baseada no contexto"""
        
        # Verifica se não é uma resposta bloqueada
        response_lower = response.lower()
        for blocked in self.blocked_responses:
            if blocked in response_lower:
                return False, "Resposta genérica não permitida", 0.0
        
        # Verifica evidência do contexto na resposta
        evidence_score = self._calculate_evidence_score(response, context_chunks)
        
        if evidence_score < self.required_evidence_threshold:
            return False, "Resposta não baseada no contexto", evidence_score
        
        # Verifica tamanho mínimo
        if len(response.strip()) < 20:
            return False, "Resposta muito curta", evidence_score
        
        return True, "OK", evidence_score
    
    def _calculate_evidence_score(self, response: str, chunks: List) -> float:
        """Calcula quanto da resposta está baseada no contexto"""
        if not chunks:
            return 0.0
        
        response_words = set(response.lower().split())
        context_words = set()
        
        for chunk in chunks:
            context_words.update(chunk.page_content.lower().split())
        
        if not response_words:
            return 0.0
        
        evidence_words = response_words.intersection(context_words)
        return len(evidence_words) / len(response_words)
```

---

## 2. Busca híbrida (vetorial + lexical)

Combina busca semântica (embeddings) com busca tradicional (palavras-chave) para melhor precisão.

### 2.1. Implementação de busca híbrida

```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, vectorstore, documents):
        self.vectorstore = vectorstore
        self.documents = documents
        
        # Prepara BM25 para busca lexical
        tokenized_docs = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Pesos para combinar scores
        self.vector_weight = 0.7
        self.bm25_weight = 0.3
    
    def hybrid_search(self, query: str, k: int = 5) -> List:
        """Busca híbrida combinando vetorial e lexical"""
        
        # 1. Busca vetorial (semântica)
        vector_results = self.vectorstore.similarity_search_with_score(query, k=k*2)
        
        # 2. Busca BM25 (lexical)
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # 3. Normaliza scores
        vector_scores = self._normalize_scores([score for _, score in vector_results])
        bm25_scores = self._normalize_scores(bm25_scores)
        
        # 4. Combina scores
        combined_results = []
        for i, (doc, vec_score) in enumerate(vector_results):
            if i < len(bm25_scores):
                # Score híbrido
                hybrid_score = (
                    self.vector_weight * vector_scores[i] + 
                    self.bm25_weight * bm25_scores[i]
                )
                
                combined_results.append((doc, hybrid_score))
        
        # 5. Ordena por score combinado e retorna top-k
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in combined_results[:k]]
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normaliza scores para 0-1"""
        if not scores:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
```

### 2.2. Retriever com fallback

```python
class FallbackRetriever:
    def __init__(self, primary_retriever, fallback_retriever):
        self.primary = primary_retriever
        self.fallback = fallback_retriever
        self.min_results = 2
    
    def retrieve_with_fallback(self, query: str, k: int = 5) -> List:
        """Busca com fallback se resultados primários forem insuficientes"""
        
        # Tenta busca primária (híbrida)
        primary_results = self.primary.hybrid_search(query, k)
        
        if len(primary_results) >= self.min_results:
            return primary_results
        
        print("⚠️  Poucos resultados na busca primária, usando fallback...")
        
        # Busca fallback (só vetorial)
        fallback_results = self.fallback.similarity_search(query, k=k*2)
        
        # Combina resultados (remove duplicatas)
        seen_content = set(doc.page_content for doc in primary_results)
        combined = list(primary_results)
        
        for doc in fallback_results:
            if doc.page_content not in seen_content and len(combined) < k:
                combined.append(doc)
                seen_content.add(doc.page_content)
        
        return combined[:k]
```

---

## 3. Reranking inteligente

Reordena resultados usando critérios mais sofisticados que apenas similaridade vetorial.

### 3.1. Reranker baseado em múltiplos critérios

```python
class MultiCriteriaReranker:
    def __init__(self):
        self.weights = {
            'semantic_similarity': 0.4,
            'keyword_match': 0.3,
            'freshness': 0.1,
            'content_quality': 0.2
        }
    
    def rerank(self, chunks: List, query: str) -> List:
        """Reordena chunks usando múltiplos critérios"""
        scored_chunks = []
        
        for chunk in chunks:
            scores = {
                'semantic_similarity': self._semantic_score(chunk, query),
                'keyword_match': self._keyword_score(chunk, query),
                'freshness': self._freshness_score(chunk),
                'content_quality': self._quality_score(chunk)
            }
            
            # Score final ponderado
            final_score = sum(
                scores[criterion] * self.weights[criterion]
                for criterion in scores
            )
            
            chunk.metadata['rerank_score'] = final_score
            chunk.metadata['score_breakdown'] = scores
            scored_chunks.append((chunk, final_score))
        
        # Ordena por score final
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in scored_chunks]
    
    def _semantic_score(self, chunk, query: str) -> float:
        """Score baseado na similaridade semântica original"""
        return chunk.metadata.get('relevance_score', 0.5)
    
    def _keyword_score(self, chunk, query: str) -> float:
        """Score baseado em correspondência de palavras-chave"""
        content_words = set(chunk.page_content.lower().split())
        query_words = set(query.lower().split())
        
        if not query_words:
            return 0.0
        
        matches = len(content_words.intersection(query_words))
        return min(matches / len(query_words), 1.0)
    
    def _freshness_score(self, chunk) -> float:
        """Score baseado na 'frescura' do documento"""
        # Implementação simplificada - pode usar timestamps reais
        source = chunk.metadata.get('source', '')
        
        # Prioriza certos tipos de documento
        if 'recent' in source or '2024' in source:
            return 1.0
        elif 'old' in source or '2020' in source:
            return 0.3
        else:
            return 0.7
    
    def _quality_score(self, chunk) -> float:
        """Score baseado na qualidade do conteúdo"""
        content = chunk.page_content
        
        # Penaliza chunks muito curtos ou muito longos
        length_score = min(len(content) / 500, 1.0) if len(content) < 1000 else 0.8
        
        # Bonus para chunks com estrutura (listas, títulos)
        structure_score = 0.0
        if ':' in content or '-' in content or '•' in content:
            structure_score = 0.2
        
        # Penaliza muito texto em maiúscula (pode ser spam)
        uppercase_ratio = sum(1 for c in content if c.isupper()) / len(content)
        uppercase_penalty = 0.0 if uppercase_ratio < 0.3 else -0.3
        
        return max(0.0, length_score + structure_score + uppercase_penalty)
```

### 3.2. Reranker baseado em LLM

```python
from langchain_ollama import ChatOllama

class LLMReranker:
    def __init__(self):
        self.llm = ChatOllama(model="llama3", temperature=0.1)
    
    def rerank_with_llm(self, chunks: List, query: str, top_k: int = 3) -> List:
        """Usa LLM para reordenar chunks por relevância"""
        
        if len(chunks) <= top_k:
            return chunks
        
        # Prepara prompt para reranking
        chunks_text = ""
        for i, chunk in enumerate(chunks):
            chunks_text += f"\n[{i+1}] {chunk.page_content[:200]}...\n"
        
        prompt = f"""
Dada a pergunta e os trechos de texto abaixo, ordene os trechos do mais relevante para o menos relevante para responder a pergunta.

PERGUNTA: {query}

TRECHOS:
{chunks_text}

Responda apenas com os números dos trechos em ordem de relevância, separados por vírgula.
Exemplo: 3,1,5,2,4

ORDEM:
"""
        
        try:
            response = self.llm.invoke(prompt)
            order = [int(x.strip()) - 1 for x in response.content.split(',')]
            
            # Reordena chunks conforme a resposta do LLM
            reranked = []
            for idx in order:
                if 0 <= idx < len(chunks):
                    reranked.append(chunks[idx])
            
            # Adiciona chunks não mencionados no final
            mentioned = set(order)
            for i, chunk in enumerate(chunks):
                if i not in mentioned:
                    reranked.append(chunk)
            
            return reranked[:top_k]
            
        except Exception as e:
            print(f"Erro no reranking com LLM: {e}")
            return chunks[:top_k]
```

---

## 4. Multi-retrieval e fusão de resultados

Combina resultados de múltiplas estratégias de busca para melhor cobertura.

### 4.1. Multi-retriever

```python
class MultiRetriever:
    def __init__(self, retrievers: dict):
        self.retrievers = retrievers
        self.fusion_method = 'rrf'  # Reciprocal Rank Fusion
    
    def multi_retrieve(self, query: str, k: int = 5) -> List:
        """Busca usando múltiplos retrievers e funde resultados"""
        
        all_results = {}
        
        # Coleta resultados de cada retriever
        for name, retriever in self.retrievers.items():
            try:
                results = retriever.get_relevant_documents(query)
                all_results[name] = results
                print(f"🔍 {name}: {len(results)} resultados")
            except Exception as e:
                print(f"❌ Erro em {name}: {e}")
                all_results[name] = []
        
        # Funde resultados
        if self.fusion_method == 'rrf':
            return self._reciprocal_rank_fusion(all_results, k)
        else:
            return self._simple_fusion(all_results, k)
    
    def _reciprocal_rank_fusion(self, results: dict, k: int) -> List:
        """Fusão usando Reciprocal Rank Fusion"""
        doc_scores = {}
        
        for retriever_name, docs in results.items():
            for rank, doc in enumerate(docs):
                doc_id = self._get_doc_id(doc)
                
                # RRF score: 1 / (rank + 60)
                rrf_score = 1 / (rank + 60)
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {'doc': doc, 'score': 0, 'sources': []}
                
                doc_scores[doc_id]['score'] += rrf_score
                doc_scores[doc_id]['sources'].append(f"{retriever_name}#{rank+1}")
        
        # Ordena por score e retorna top-k
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        result_docs = []
        for item in sorted_docs[:k]:
            doc = item['doc']
            doc.metadata['fusion_score'] = item['score']
            doc.metadata['fusion_sources'] = item['sources']
            result_docs.append(doc)
        
        return result_docs
    
    def _get_doc_id(self, doc) -> str:
        """Gera ID único para o documento"""
        return hash(doc.page_content[:100])
    
    def _simple_fusion(self, results: dict, k: int) -> List:
        """Fusão simples por concatenação"""
        all_docs = []
        seen_content = set()
        
        # Intercala resultados de cada retriever
        max_len = max(len(docs) for docs in results.values()) if results else 0
        
        for i in range(max_len):
            for retriever_name, docs in results.items():
                if i < len(docs):
                    doc = docs[i]
                    content_hash = hash(doc.page_content)
                    
                    if content_hash not in seen_content:
                        doc.metadata['retriever_source'] = retriever_name
                        all_docs.append(doc)
                        seen_content.add(content_hash)
                        
                        if len(all_docs) >= k:
                            return all_docs
        
        return all_docs
```

---

## 5. Validação e filtragem de respostas

### 5.1. Sistema de validação em cascata

```python
class ResponseValidationPipeline:
    def __init__(self):
        self.validators = [
            self._validate_length,
            self._validate_relevance,
            self._validate_safety,
            self._validate_coherence
        ]
    
    def validate_response(self, response: str, query: str, context: List) -> dict:
        """Executa pipeline de validação completo"""
        
        validation_result = {
            'is_valid': True,
            'confidence': 1.0,
            'issues': [],
            'scores': {}
        }
        
        for validator in self.validators:
            result = validator(response, query, context)
            
            validation_result['scores'][result['name']] = result['score']
            
            if not result['passed']:
                validation_result['is_valid'] = False
                validation_result['issues'].append(result['issue'])
            
            # Reduz confiança baseado no score
            validation_result['confidence'] *= result['score']
        
        return validation_result
    
    def _validate_length(self, response: str, query: str, context: List) -> dict:
        """Valida se a resposta tem tamanho adequado"""
        length = len(response.strip())
        
        if length < 20:
            return {
                'name': 'length',
                'passed': False,
                'score': 0.2,
                'issue': 'Resposta muito curta'
            }
        elif length > 2000:
            return {
                'name': 'length',
                'passed': False,
                'score': 0.8,
                'issue': 'Resposta muito longa'
            }
        else:
            score = min(1.0, length / 500)  # Ideal entre 100-500 chars
            return {
                'name': 'length',
                'passed': True,
                'score': score,
                'issue': None
            }
    
    def _validate_relevance(self, response: str, query: str, context: List) -> dict:
        """Valida relevância da resposta para a pergunta"""
        # Implementação simplificada usando sobreposição de palavras
        response_words = set(response.lower().split())
        query_words = set(query.lower().split())
        
        overlap = len(response_words.intersection(query_words))
        relevance_score = overlap / len(query_words) if query_words else 0
        
        return {
            'name': 'relevance',
            'passed': relevance_score > 0.1,
            'score': min(relevance_score * 2, 1.0),
            'issue': 'Resposta não relacionada à pergunta' if relevance_score <= 0.1 else None
        }
    
    def _validate_safety(self, response: str, query: str, context: List) -> dict:
        """Valida se a resposta é segura"""
        unsafe_patterns = [
            r'\b(hack|crack|bypass|exploit)\b',
            r'\b(delete|remove|destroy)\b.*\b(file|data|system)\b',
            r'\bpersonal\s+information\b'
        ]
        
        response_lower = response.lower()
        
        for pattern in unsafe_patterns:
            if re.search(pattern, response_lower):
                return {
                    'name': 'safety',
                    'passed': False,
                    'score': 0.0,
                    'issue': 'Resposta potencialmente insegura'
                }
        
        return {
            'name': 'safety',
            'passed': True,
            'score': 1.0,
            'issue': None
        }
    
    def _validate_coherence(self, response: str, query: str, context: List) -> dict:
        """Valida coerência da resposta"""
        # Verifica se a resposta tem estrutura básica
        sentences = response.split('.')
        
        if len(sentences) < 1:
            score = 0.3
        elif len(sentences) == 1:
            score = 0.7
        else:
            score = 1.0
        
        # Verifica repetições excessivas
        words = response.lower().split()
        unique_words = set(words)
        repetition_ratio = len(unique_words) / len(words) if words else 1
        
        if repetition_ratio < 0.5:
            score *= 0.5
        
        return {
            'name': 'coherence',
            'passed': score > 0.5,
            'score': score,
            'issue': 'Resposta incoerente ou repetitiva' if score <= 0.5 else None
        }
```

---

## 6. Memória conversacional

### 6.1. Gerenciador de histórico

```python
class ConversationMemory:
    def __init__(self, max_history=10):
        self.conversations = {}
        self.max_history = max_history
    
    def add_exchange(self, session_id: str, question: str, answer: str, context: List):
        """Adiciona uma troca de conversa ao histórico"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        exchange = {
            'timestamp': time.time(),
            'question': question,
            'answer': answer,
            'context_sources': [chunk.metadata.get('source') for chunk in context]
        }
        
        self.conversations[session_id].append(exchange)
        
        # Limita tamanho do histórico
        if len(self.conversations[session_id]) > self.max_history:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history:]
    
    def get_context_aware_query(self, session_id: str, current_query: str) -> str:
        """Reformula a consulta considerando o contexto conversacional"""
        
        if session_id not in self.conversations or not self.conversations[session_id]:
            return current_query
        
        recent_exchanges = self.conversations[session_id][-3:]  # Últimas 3 trocas
        
        # Verifica se a consulta atual é uma referência (pronomes, etc.)
        reference_patterns = [
            r'\b(isso|isto|aquilo|essa|esta)\b',
            r'\b(ele|ela|eles|elas)\b',
            r'\b(como|onde|quando)\s+(mesmo|também)\b'
        ]
        
        has_reference = any(re.search(pattern, current_query, re.IGNORECASE) 
                          for pattern in reference_patterns)
        
        if has_reference and recent_exchanges:
            # Adiciona contexto da última pergunta
            last_question = recent_exchanges[-1]['question']
            enhanced_query = f"Contexto anterior: {last_question}\nPergunta atual: {current_query}"
            return enhanced_query
        
        return current_query
    
    def get_conversation_summary(self, session_id: str) -> str:
        """Gera resumo da conversa para contexto"""
        if session_id not in self.conversations:
            return ""
        
        exchanges = self.conversations[session_id]
        if not exchanges:
            return ""
        
        summary_lines = []
        for exchange in exchanges[-5:]:  # Últimas 5 trocas
            summary_lines.append(f"P: {exchange['question'][:100]}...")
            summary_lines.append(f"R: {exchange['answer'][:100]}...")
        
        return "\n".join(summary_lines)
```

---

## 7. Métricas e monitoramento avançado

### 7.1. Sistema de métricas detalhado

```python
import json
from datetime import datetime, timedelta

class AdvancedMetrics:
    def __init__(self):
        self.metrics = {
            'queries': [],
            'performance': {},
            'quality': {},
            'errors': []
        }
    
    def log_query(self, query_data: dict):
        """Registra métricas detalhadas de uma consulta"""
        query_data['timestamp'] = time.time()
        self.metrics['queries'].append(query_data)
    
    def log_error(self, error_type: str, error_msg: str, context: dict = None):
        """Registra erros para análise"""
        error_data = {
            'timestamp': time.time(),
            'type': error_type,
            'message': error_msg,
            'context': context or {}
        }
        self.metrics['errors'].append(error_data)
    
    def calculate_quality_metrics(self, time_window_hours: int = 24) -> dict:
        """Calcula métricas de qualidade no período"""
        cutoff_time = time.time() - (time_window_hours * 3600)
        recent_queries = [q for q in self.metrics['queries'] if q['timestamp'] >= cutoff_time]
        
        if not recent_queries:
            return {}
        
        # Métricas de qualidade
        quality_metrics = {
            'total_queries': len(recent_queries),
            'avg_response_time': sum(q.get('response_time', 0) for q in recent_queries) / len(recent_queries),
            'avg_chunks_retrieved': sum(q.get('chunks_count', 0) for q in recent_queries) / len(recent_queries),
            'validation_pass_rate': sum(1 for q in recent_queries if q.get('validation_passed', False)) / len(recent_queries),
            'avg_confidence': sum(q.get('confidence', 0) for q in recent_queries) / len(recent_queries)
        }
        
        # Métricas de erro
        recent_errors = [e for e in self.metrics['errors'] if e['timestamp'] >= cutoff_time]
        quality_metrics['error_rate'] = len(recent_errors) / len(recent_queries) if recent_queries else 0
        
        return quality_metrics
    
    def get_performance_report(self) -> dict:
        """Gera relatório de performance detalhado"""
        recent_metrics = self.calculate_quality_metrics()
        
        # Análise de tendências
        if len(self.metrics['queries']) >= 10:
            recent_10 = self.metrics['queries'][-10:]
            previous_10 = self.metrics['queries'][-20:-10] if len(self.metrics['queries']) >= 20 else []
            
            trend_data = {}
            if previous_10:
                recent_avg_time = sum(q.get('response_time', 0) for q in recent_10) / len(recent_10)
                previous_avg_time = sum(q.get('response_time', 0) for q in previous_10) / len(previous_10)
                
                trend_data['response_time_trend'] = (recent_avg_time - previous_avg_time) / previous_avg_time if previous_avg_time > 0 else 0
            
            recent_metrics['trends'] = trend_data
        
        return recent_metrics
    
    def export_metrics(self, filepath: str):
        """Exporta métricas para arquivo"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
```

---

## 8. Otimizações para produção

### 8.1. Cache inteligente

```python
from functools import lru_cache
import hashlib

class IntelligentCache:
    def __init__(self, max_size=1000, ttl_hours=24):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600
    
    def _get_cache_key(self, query: str, context_hash: str = "") -> str:
        """Gera chave de cache baseada na consulta e contexto"""
        combined = f"{query}|{context_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, query: str, context_hash: str = "") -> dict:
        """Recupera resposta do cache se disponível e válida"""
        key = self._get_cache_key(query, context_hash)
        
        if key in self.cache:
            cached_item = self.cache[key]
            
            # Verifica TTL
            if time.time() - cached_item['timestamp'] < self.ttl_seconds:
                self.access_times[key] = time.time()  # Atualiza acesso
                return cached_item['data']
            else:
                # Remove item expirado
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
        
        return None
    
    def set(self, query: str, response_data: dict, context_hash: str = ""):
        """Armazena resposta no cache"""
        key = self._get_cache_key(query, context_hash)
        
        # Remove itens antigos se necessário
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = {
            'timestamp': time.time(),
            'data': response_data
        }
        self.access_times[key] = time.time()
    
    def _evict_oldest(self):
        """Remove o item menos recentemente usado"""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
```

### 8.2. Pool de conexões e rate limiting

```python
import asyncio
from asyncio import Semaphore
import time

class ResourceManager:
    def __init__(self, max_concurrent=10, rate_limit_per_minute=100):
        self.semaphore = Semaphore(max_concurrent)
        self.rate_limit = rate_limit_per_minute
        self.requests_timestamps = []
    
    async def execute_with_limits(self, func, *args, **kwargs):
        """Executa função respeitando limites de concorrência e rate limiting"""
        
        # Verifica rate limit
        await self._check_rate_limit()
        
        # Controla concorrência
        async with self.semaphore:
            return await func(*args, **kwargs)
    
    async def _check_rate_limit(self):
        """Verifica e aplica rate limiting"""
        current_time = time.time()
        
        # Remove timestamps antigos (mais de 1 minuto)
        self.requests_timestamps = [
            ts for ts in self.requests_timestamps 
            if current_time - ts < 60
        ]
        
        # Verifica se excedeu o limite
        if len(self.requests_timestamps) >= self.rate_limit:
            # Calcula tempo de espera
            oldest_request = min(self.requests_timestamps)
            wait_time = 60 - (current_time - oldest_request)
            
            if wait_time > 0:
                print(f"⏳ Rate limit atingido, aguardando {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
        
        # Registra nova requisição
        self.requests_timestamps.append(current_time)
```

---

## 9. Conclusão

Implementamos um sistema RAG avançado com:

- **Guardrails robustos:** validação em múltiplas camadas
- **Busca híbrida:** combinação de semântica e lexical  
- **Reranking inteligente:** múltiplos critérios e LLM
- **Multi-retrieval:** fusão de estratégias diferentes
- **Validação de qualidade:** pipeline de verificação
- **Memória conversacional:** contexto entre perguntas
- **Monitoramento:** métricas detalhadas e relatórios
- **Otimizações:** cache, rate limiting e performance

Nosso RAG agora está pronto para ambientes de produção, com controles de qualidade, performance otimizada e observabilidade completa.

No próximo capítulo, vamos ver como implementar avaliação automática e testes para garantir que o sistema mantenha alta qualidade ao longo do tempo.

**Exercício prático:** implemente o sistema de guardrails no seu projeto e teste com consultas problemáticas para ver as validações em ação.

---

### Pergunta ao leitor

Agora que temos um RAG avançado e robusto, quer que eu escreva o **Capítulo 07 — Avaliação e Testes de Sistemas RAG** focando em métricas automáticas e benchmarking?
