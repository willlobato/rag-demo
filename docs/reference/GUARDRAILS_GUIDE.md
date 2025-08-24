# 🛡️ Guia Completo de Guardrails para Sistemas RAG

> **Documentação técnica avançada sobre implementação de guardrails**  
> **Para sistemas RAG seguros, confiáveis e livres de alucinações**

---

## 📚 **ÍNDICE**

1. **[Introdução aos Guardrails](#introdução-aos-guardrails)**
2. **[Arquitetura de 4 Camadas](#arquitetura-de-4-camadas)**
3. **[Implementação Técnica](#implementação-técnica)**
4. **[Configuração de Thresholds](#configuração-de-thresholds)**
5. **[Templates de Prompt](#templates-de-prompt)**
6. **[Métricas e Monitoramento](#métricas-e-monitoramento)**
7. **[Casos de Uso Avançados](#casos-de-uso-avançados)**
8. **[Troubleshooting](#troubleshooting)**

---

## 🎯 **Introdução aos Guardrails**

### **Definição**
Guardrails são **sistemas de controle e validação** que garantem que sistemas RAG produzam respostas:
- **Factualmente corretas** (baseadas no contexto)
- **Seguras** (sem conteúdo malicioso)
- **Confiáveis** (sem alucinações)
- **Rastreáveis** (com citação de fontes)

### **Por que são Essenciais?**

**Problemas Sem Guardrails:**
```
❌ Query: "Qual o CEO da empresa?"
❌ Resposta: "O CEO atual é John Smith, assumiu em 2023..."
❌ Problema: Informação INVENTADA - não está nos documentos
```

**Com Guardrails:**
```
✅ Query: "Qual o CEO da empresa?"
✅ Resposta: "❌ Não encontrei informações sobre CEO no contexto disponível."
✅ Resultado: HONESTIDADE - sistema admite limitações
```

### **Tipos de Falhas que Guardrails Previnem**

1. **Alucinações Factuais**
   - LLM inventa fatos não presentes no contexto
   - Dados numéricos incorretos
   - Nomes, datas, locais inventados

2. **Extrapolação Inadequada**
   - LLM faz inferências além do contexto
   - Conclusões não suportadas pelos dados
   - Generalizações excessivas

3. **Ataques de Injection**
   - Tentativas de manipular o prompt
   - Bypass de instruções de sistema
   - Extração de informações sensíveis

4. **Respostas Inconsistentes**
   - Variação entre execuções
   - Contradições internas
   - Falta de reprodutibilidade

---

## 🏗️ **Arquitetura de 4 Camadas**

### **Fluxo Completo com Guardrails**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   INPUT QUERY   │───▶│ 🔍 INPUT GUARDS │───▶│  VALIDATED QUERY │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ FILTERED CHUNKS │◀───│⚖️ RETRIEVAL     │◀───│   VECTOR SEARCH  │
└─────────────────┘    │   GUARDS        │    └─────────────────┘
                       └─────────────────┘
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  LLM RESPONSE   │◀───│📝 PROMPT GUARDS │◀───│  CONTEXT PREP   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ FINAL RESPONSE  │◀───│✅ OUTPUT GUARDS │◀───│  RAW RESPONSE   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **1. INPUT GUARDRAILS (Primeira Linha de Defesa)**

**Validações Implementadas:**
```python
def validate_input(query: str) -> Tuple[bool, str, str]:
    # Validação básica
    if len(query.strip()) < 3:
        return False, "", "Query muito curta"
    
    # Detecção de injection
    injection_patterns = [
        "ignore previous instructions",
        "forget everything", 
        "act as",
        "system:",
        "pretend to be"
    ]
    
    query_lower = query.lower()
    for pattern in injection_patterns:
        if pattern in query_lower:
            return False, query, f"Injection detectada: {pattern}"
    
    return True, query.strip(), ""
```

**Casos Bloqueados:**
- Queries vazias ou muito curtas
- Tentativas de prompt injection
- Caracteres especiais maliciosos
- Comandos de sistema embebidos

### **2. RETRIEVAL GUARDRAILS (Controle de Relevância)**

**Threshold de Similaridade:**
```python
def filter_by_threshold(results: List[Tuple[Document, float]], 
                       threshold: float) -> List[Document]:
    """
    Filtra chunks por similaridade usando threshold rigoroso.
    
    ChromaDB retorna distância (menor = mais similar):
    - Score 0.15 = muito similar (ACEITAR)
    - Score 0.45 = pouco similar (REJEITAR se threshold=0.35)
    """
    filtered = []
    for doc, score in results:
        if score <= threshold:  # Menor distância = mais similar
            if validate_chunk_quality(doc):
                filtered.append(doc)
                log_acceptance(doc, score)
            else:
                log_quality_rejection(doc, score)
        else:
            log_threshold_rejection(doc, score, threshold)
    
    return filtered
```

**Validação de Qualidade de Chunks:**
```python
def validate_chunk_quality(doc: Document) -> bool:
    content = doc.page_content.strip()
    
    # Muito pequeno
    if len(content) < 50:
        return False
    
    # Apenas pontuação/números
    alpha_ratio = sum(c.isalpha() for c in content) / len(content)
    if alpha_ratio < 0.5:
        return False
    
    # Texto repetitivo (heurística)
    words = content.split()
    if len(set(words)) / len(words) < 0.3:  # Muita repetição
        return False
    
    return True
```

**Estratégias de Threshold:**
```python
# Configurações por tipo de sistema
THRESHOLD_CONFIGS = {
    "critical": 0.20,      # Sistema crítico (medicina, finanças)
    "technical": 0.30,     # Documentação técnica
    "support": 0.40,       # FAQ/Suporte
    "exploratory": 0.50    # Busca exploratória
}
```

### **3. PROMPT GUARDRAILS (Controle de Comportamento)**

**Template Rigoroso (STRICT):**
```python
STRICT_TEMPLATE = """
Você é um assistente que responde EXCLUSIVAMENTE com base no contexto fornecido.

REGRAS OBRIGATÓRIAS (NÃO QUEBRE NUNCA):
1. Use APENAS informações presentes no CONTEXTO abaixo
2. Se a resposta não estiver no contexto, responda EXATAMENTE: 
   "❌ Não encontrei informações relevantes no contexto disponível."
3. NUNCA invente, deduza, extrapole ou use conhecimento externo
4. SEMPRE cite a fonte: (Fonte: nome_do_arquivo)
5. Responda em português brasileiro, seja objetivo e direto

FORMATO OBRIGATÓRIO:
✅ Com base no contexto fornecido: [SUA_RESPOSTA_AQUI]
(Fonte: nome_do_arquivo)

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA:
"""
```

**Template Balanceado (BALANCED):**
```python
BALANCED_TEMPLATE = """
Você é um assistente especializado que prioriza informações do contexto fornecido.

DIRETRIZES:
1. PRIORIZE SEMPRE as informações do CONTEXTO
2. Use conhecimento geral APENAS para esclarecimentos básicos
3. INDIQUE CLARAMENTE a origem das informações:
   - "Com base no contexto:" para info do contexto
   - "Informação complementar:" para conhecimento geral
4. Se não houver contexto suficiente, seja transparente
5. Sempre cite fontes quando disponíveis

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA:
"""
```

**Técnicas Avançadas de Prompt:**
```python
def build_context_aware_prompt(context: str, question: str, 
                              context_confidence: float) -> str:
    """Adapta template baseado na confiança do contexto."""
    
    if context_confidence < 0.3:
        # Contexto de baixa qualidade - template muito rigoroso
        return STRICT_TEMPLATE.format(context=context, question=question)
    
    elif context_confidence > 0.8:
        # Contexto de alta qualidade - template mais flexível
        return BALANCED_TEMPLATE.format(context=context, question=question)
    
    else:
        # Contexto médio - template padrão com avisos
        enhanced_template = STRICT_TEMPLATE.replace(
            "Com base no contexto fornecido:",
            "⚠️ Com base no contexto de qualidade média fornecido:"
        )
        return enhanced_template.format(context=context, question=question)
```

### **4. OUTPUT GUARDRAILS (Validação Final)**

**Validação de Fidelidade:**
```python
def validate_response_fidelity(response: str, context: str, 
                              threshold: float = 0.3) -> Tuple[bool, float]:
    """
    Valida se resposta está fiel ao contexto usando overlap de palavras.
    """
    # Tokenização e limpeza
    context_words = set(re.findall(r'\w+', context.lower()))
    response_words = set(re.findall(r'\w+', response.lower()))
    
    # Remover stopwords
    stopwords = {'o', 'a', 'de', 'da', 'do', 'para', 'com', 'em', 'que', 'é'}
    context_words -= stopwords
    response_words -= stopwords
    
    # Calcular overlap
    if len(response_words) == 0:
        return False, 0.0
    
    overlap = len(context_words.intersection(response_words))
    fidelity_score = overlap / len(response_words)
    
    return fidelity_score >= threshold, fidelity_score
```

**Detecção de Alucinação:**
```python
def detect_hallucination(response: str, context: str, 
                        question: str) -> Dict[str, Any]:
    """Detecta possíveis alucinações na resposta."""
    
    detection_result = {
        "has_hallucination": False,
        "confidence": 0.0,
        "reasons": []
    }
    
    # 1. Verifica informações numéricas específicas
    response_numbers = re.findall(r'\d+', response)
    context_numbers = re.findall(r'\d+', context)
    
    for num in response_numbers:
        if num not in context_numbers:
            detection_result["reasons"].append(f"Número {num} não encontrado no contexto")
            detection_result["has_hallucination"] = True
    
    # 2. Verifica nomes próprios
    response_proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', response)
    context_proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', context)
    
    for noun in response_proper_nouns:
        if noun not in context_proper_nouns and noun not in question:
            detection_result["reasons"].append(f"Nome próprio {noun} não no contexto")
            detection_result["has_hallucination"] = True
    
    # 3. Verifica citação de fonte
    has_source_citation = any(pattern in response.lower() for pattern in 
                             ["fonte:", "(fonte", "baseado no contexto"])
    
    if not has_source_citation and "❌ Não encontrei" not in response:
        detection_result["reasons"].append("Falta citação de fonte")
        detection_result["has_hallucination"] = True
    
    # 4. Calcula confiança final
    if detection_result["has_hallucination"]:
        detection_result["confidence"] = min(1.0, len(detection_result["reasons"]) * 0.3)
    
    return detection_result
```

---

## ⚙️ **Configuração de Thresholds**

### **Metodologia de Otimização**

**1. Análise Estatística Base:**
```python
def analyze_similarity_distribution(vectorstore, test_queries):
    """Analisa distribuição de scores para encontrar thresholds naturais."""
    all_scores = []
    
    for query in test_queries:
        results = vectorstore.similarity_search_with_score(query, k=20)
        scores = [score for _, score in results]
        all_scores.extend(scores)
    
    stats = {
        "mean": np.mean(all_scores),
        "std": np.std(all_scores),
        "median": np.median(all_scores),
        "q1": np.percentile(all_scores, 25),
        "q3": np.percentile(all_scores, 75),
        "p90": np.percentile(all_scores, 90),
        "p95": np.percentile(all_scores, 95)
    }
    
    return stats, all_scores
```

**2. Sugestões Automáticas:**
```python
def suggest_thresholds(stats):
    """Sugere thresholds baseado em análise estatística."""
    return {
        "conservative": stats["mean"] - 2 * stats["std"],    # Top 2-5%
        "strict": stats["q1"],                               # Top 25%
        "balanced": stats["median"],                         # Top 50%
        "permissive": stats["q3"],                          # Top 75%
        "exploratory": stats["p90"]                         # Top 90%
    }
```

**3. Teste Empírico:**
```python
def evaluate_threshold_performance(thresholds, test_queries, golden_answers=None):
    """Testa performance de diferentes thresholds."""
    results = {}
    
    for threshold in thresholds:
        metrics = {
            "acceptance_rate": 0,
            "avg_chunks_used": 0,
            "processing_time": 0,
            "quality_score": 0  # Se tiver golden answers
        }
        
        for query in test_queries:
            start_time = time.time()
            
            # Executar RAG com threshold
            response = rag_with_threshold(query, threshold)
            
            metrics["processing_time"] += time.time() - start_time
            
            if response["status"] == "success":
                metrics["acceptance_rate"] += 1
                metrics["avg_chunks_used"] += len(response["chunks_used"])
                
                # Avaliar qualidade se tiver resposta de referência
                if golden_answers and query in golden_answers:
                    quality = calculate_quality(response["answer"], golden_answers[query])
                    metrics["quality_score"] += quality
        
        # Normalizar métricas
        metrics["acceptance_rate"] /= len(test_queries)
        metrics["avg_chunks_used"] /= len(test_queries)
        metrics["processing_time"] /= len(test_queries)
        metrics["quality_score"] /= len(test_queries)
        
        results[threshold] = metrics
    
    return results
```

### **Configuração por Domínio**

**Documentação Técnica (nosso caso):**
```python
TECHNICAL_CONFIG = {
    "similarity_threshold": 0.30,
    "min_chunks_required": 1,
    "max_chunks_to_llm": 4,
    "template_mode": "strict",
    "require_source_citation": True,
    "fidelity_threshold": 0.4
}
```

**Sistema Crítico (medicina, finanças):**
```python
CRITICAL_CONFIG = {
    "similarity_threshold": 0.20,
    "min_chunks_required": 2,
    "max_chunks_to_llm": 3,
    "template_mode": "strict",
    "require_source_citation": True,
    "fidelity_threshold": 0.6,
    "enable_human_review": True
}
```

**FAQ/Suporte:**
```python
SUPPORT_CONFIG = {
    "similarity_threshold": 0.45,
    "min_chunks_required": 1,
    "max_chunks_to_llm": 6,
    "template_mode": "balanced",
    "require_source_citation": False,
    "fidelity_threshold": 0.3
}
```

---

## 📊 **Métricas e Monitoramento**

### **Métricas de Guardrails**

**1. Métricas de Input:**
```python
input_metrics = {
    "total_queries": 1000,
    "valid_queries": 950,
    "rejected_injection": 30,
    "rejected_format": 20,
    "validation_rate": 0.95
}
```

**2. Métricas de Retrieval:**
```python
retrieval_metrics = {
    "avg_chunks_retrieved": 8.5,
    "avg_chunks_accepted": 3.2,
    "acceptance_rate": 0.376,  # 37.6% dos chunks passam no threshold
    "avg_similarity_score": 0.34,
    "queries_with_no_context": 45,  # 4.5% das queries
    "no_context_rate": 0.045
}
```

**3. Métricas de Generation:**
```python
generation_metrics = {
    "responses_generated": 905,  # queries que chegaram à geração
    "avg_response_length": 156,
    "avg_generation_time": 2.3,  # segundos
    "template_strict_usage": 0.7,
    "template_balanced_usage": 0.3
}
```

**4. Métricas de Output:**
```python
output_metrics = {
    "responses_validated": 905,
    "responses_with_source": 890,  # 98.3%
    "avg_fidelity_score": 0.67,
    "hallucination_detected": 15,  # 1.7%
    "hallucination_rate": 0.017,
    "final_success_rate": 0.882
}
```

### **Dashboard de Monitoramento**

**Visualização em Tempo Real:**
```python
def create_guardrails_dashboard():
    """Cria dashboard de monitoramento de guardrails."""
    
    metrics = collect_realtime_metrics()
    
    dashboard = {
        "overview": {
            "total_queries_today": metrics["input"]["total_queries"],
            "success_rate": metrics["output"]["final_success_rate"],
            "avg_response_time": metrics["generation"]["avg_generation_time"],
            "current_threshold": get_current_threshold()
        },
        
        "quality_gates": {
            "input_validation": metrics["input"]["validation_rate"],
            "context_filtering": metrics["retrieval"]["acceptance_rate"],
            "fidelity_check": metrics["output"]["avg_fidelity_score"],
            "hallucination_rate": metrics["output"]["hallucination_rate"]
        },
        
        "performance": {
            "queries_per_minute": calculate_qpm(),
            "p95_response_time": calculate_p95_latency(),
            "error_rate": calculate_error_rate(),
            "memory_usage": get_memory_usage()
        },
        
        "alerts": generate_alerts(metrics)
    }
    
    return dashboard
```

**Alertas Automáticos:**
```python
def generate_alerts(metrics):
    """Gera alertas baseado em thresholds de qualidade."""
    alerts = []
    
    # Taxa de sucesso muito baixa
    if metrics["output"]["final_success_rate"] < 0.8:
        alerts.append({
            "severity": "high",
            "message": f"Taxa de sucesso baixa: {metrics['output']['final_success_rate']:.1%}",
            "action": "Revisar threshold de similaridade"
        })
    
    # Muita rejeição de contexto
    if metrics["retrieval"]["no_context_rate"] > 0.1:
        alerts.append({
            "severity": "medium", 
            "message": f"Alta taxa sem contexto: {metrics['retrieval']['no_context_rate']:.1%}",
            "action": "Considerar threshold mais permissivo"
        })
    
    # Detecção de alucinações frequente
    if metrics["output"]["hallucination_rate"] > 0.05:
        alerts.append({
            "severity": "high",
            "message": f"Taxa de alucinação alta: {metrics['output']['hallucination_rate']:.1%}",
            "action": "Revisar template de prompt e threshold"
        })
    
    return alerts
```

---

## 🎛️ **Casos de Uso Avançados**

### **1. Threshold Adaptativo**

**Threshold Baseado na Complexidade da Query:**
```python
def adaptive_threshold(query: str, base_threshold: float = 0.35) -> float:
    """Ajusta threshold baseado na complexidade da query."""
    
    # Análise da query
    word_count = len(query.split())
    has_technical_terms = any(term in query.lower() for term in 
                             ['api', 'sistema', 'arquitetura', 'performance'])
    is_specific = any(char in query for char in '?')
    
    # Ajustes
    threshold = base_threshold
    
    # Queries complexas precisam de mais contexto
    if word_count > 8:
        threshold += 0.1
    
    # Queries técnicas podem ser mais rigorosas
    if has_technical_terms:
        threshold -= 0.05
    
    # Perguntas específicas podem ser mais rigorosas
    if is_specific:
        threshold -= 0.05
    
    return max(0.15, min(0.6, threshold))  # Limitar entre 0.15 e 0.6
```

**Threshold Baseado no Histórico:**
```python
def historical_threshold(user_id: str, base_threshold: float) -> float:
    """Ajusta threshold baseado no histórico de satisfação do usuário."""
    
    user_history = get_user_history(user_id)
    
    if not user_history:
        return base_threshold
    
    # Calcular satisfação média
    avg_satisfaction = np.mean([h["satisfaction"] for h in user_history])
    
    # Ajustar threshold baseado na satisfação
    if avg_satisfaction < 3.0:  # Usuário insatisfeito
        return base_threshold - 0.1  # Ser mais rigoroso
    elif avg_satisfaction > 4.0:  # Usuário satisfeito
        return base_threshold + 0.05  # Ser mais permissivo
    
    return base_threshold
```

### **2. Validação Semântica Avançada**

**Detecção de Inconsistência Semântica:**
```python
def semantic_consistency_check(response: str, context: str, 
                              embeddings_model) -> float:
    """Verifica consistência semântica entre resposta e contexto."""
    
    # Gerar embeddings
    response_embedding = embeddings_model.embed_query(response)
    context_embedding = embeddings_model.embed_query(context)
    
    # Calcular similaridade
    similarity = cosine_similarity([response_embedding], [context_embedding])[0][0]
    
    return similarity
```

**Detecção de Contradições:**
```python
def detect_contradictions(response: str, context: str) -> List[str]:
    """Detecta contradições entre resposta e contexto."""
    contradictions = []
    
    # Extrair declarações numéricas
    response_numbers = extract_numeric_statements(response)
    context_numbers = extract_numeric_statements(context)
    
    for resp_stmt in response_numbers:
        for ctx_stmt in context_numbers:
            if are_contradictory(resp_stmt, ctx_stmt):
                contradictions.append(f"Contradição: '{resp_stmt}' vs '{ctx_stmt}'")
    
    return contradictions

def extract_numeric_statements(text: str) -> List[str]:
    """Extrai declarações que contêm números."""
    sentences = sent_tokenize(text)
    numeric_sentences = []
    
    for sentence in sentences:
        if re.search(r'\d+', sentence):
            numeric_sentences.append(sentence.strip())
    
    return numeric_sentences
```

### **3. Guardrails Específicos por Domínio**

**Validação para Documentação Técnica:**
```python
def technical_domain_validation(response: str, query: str) -> Dict[str, Any]:
    """Validações específicas para documentação técnica."""
    
    validation = {
        "valid": True,
        "warnings": [],
        "errors": []
    }
    
    # 1. Verificar se métricas têm unidades
    metrics_pattern = r'(\d+(?:\.\d+)?)\s*(ms|segundos?|GB|MB|%|usuários?)'
    metrics_without_units = re.findall(r'\d+(?:\.\d+)?(?!\s*(?:ms|segundos?|GB|MB|%|usuários?))', response)
    
    if "performance" in query.lower() and metrics_without_units:
        validation["warnings"].append("Métricas sem unidades detectadas")
    
    # 2. Verificar consistência de tecnologias mencionadas
    known_technologies = {
        'spring boot', 'docker', 'kubernetes', 'postgresql', 'redis',
        'prometheus', 'grafana', 'jenkins', 'elasticsearch'
    }
    
    mentioned_techs = set()
    for tech in known_technologies:
        if tech in response.lower():
            mentioned_techs.add(tech)
    
    # Verificar se tecnologias mencionadas fazem sentido juntas
    incompatible_combinations = [
        ({'mysql', 'postgresql'}, "Menção de múltiplos SGBDs"),
        ({'docker', 'kubernetes'}, None)  # Essa é compatível
    ]
    
    for incompatible_set, message in incompatible_combinations:
        if message and incompatible_set.issubset(mentioned_techs):
            validation["warnings"].append(message)
    
    return validation
```

**Validação para Sistema de Suporte:**
```python
def support_domain_validation(response: str, query: str) -> Dict[str, Any]:
    """Validações específicas para sistema de suporte."""
    
    validation = {
        "valid": True,
        "suggestions": []
    }
    
    # 1. Verificar se resposta tem tom adequado para suporte
    if any(word in response.lower() for word in ['impossível', 'não pode', 'nunca']):
        validation["suggestions"].append("Considere tom mais positivo para suporte")
    
    # 2. Verificar se oferece próximos passos
    if '?' in query and not any(phrase in response.lower() for phrase in 
                               ['você pode', 'tente', 'próximo passo', 'contacte']):
        validation["suggestions"].append("Considere incluir próximos passos")
    
    # 3. Verificar se resposta é suficientemente detalhada
    if len(response.split()) < 10:
        validation["suggestions"].append("Resposta pode estar muito curta")
    
    return validation
```

---

## 🔧 **Troubleshooting**

### **Problemas Comuns e Soluções**

**1. Alta Taxa de Rejeição (>20%)**
```
Sintoma: Muitas queries resultam em "Não encontrei informações relevantes"
Causa: Threshold muito rigoroso
Solução:
- Executar threshold_optimizer.py para encontrar valor ideal
- Aumentar threshold gradualmente (ex: 0.30 → 0.35 → 0.40)
- Verificar qualidade dos embeddings
- Considerar re-indexação com chunks diferentes
```

**2. Respostas com Baixa Fidelidade**
```
Sintoma: Fidelity score < 0.4 consistentemente
Causa: Template muito permissivo ou threshold muito alto
Solução:
- Usar template strict em vez de balanced
- Diminuir threshold de similaridade
- Revisar instruções do prompt
- Implementar validação semântica adicional
```

**3. Tempo de Resposta Alto**
```
Sintoma: Tempo médio > 5 segundos
Causa: Muitos chunks sendo processados ou validações complexas
Solução:
- Reduzir max_chunks_to_llm
- Implementar cache de embeddings
- Otimizar validações de output
- Usar modelo de embedding menor
```

**4. Detecção Excessiva de Alucinações**
```
Sintoma: >10% de respostas marcadas como alucinação
Causa: Validação muito sensível ou dados inconsistentes
Solução:
- Ajustar thresholds de detecção
- Revisar qualidade dos dados originais
- Melhorar algoritmo de detecção
- Validar manualmente casos flagados
```

### **Debugging Avançado**

**1. Análise de Decisões de Guardrail:**
```python
def analyze_guardrail_decisions(log_file: str):
    """Analisa logs para identificar padrões nas decisões."""
    
    with open(log_file, 'r') as f:
        logs = f.readlines()
    
    decisions = {
        "input_rejected": [],
        "threshold_rejected": [],
        "quality_rejected": [],
        "hallucination_detected": []
    }
    
    for line in logs:
        if "input_rejected" in line:
            decisions["input_rejected"].append(parse_log_line(line))
        elif "threshold_rejected" in line:
            decisions["threshold_rejected"].append(parse_log_line(line))
        # ... etc
    
    # Análise de padrões
    patterns = {
        "most_rejected_query_types": analyze_query_patterns(decisions["threshold_rejected"]),
        "common_hallucination_triggers": analyze_hallucination_patterns(decisions["hallucination_detected"]),
        "threshold_effectiveness": calculate_threshold_effectiveness(decisions)
    }
    
    return patterns
```

**2. Teste A/B de Configurações:**
```python
def ab_test_guardrail_configs(config_a: dict, config_b: dict, 
                             test_queries: List[str]) -> Dict[str, Any]:
    """Testa duas configurações de guardrail para comparar performance."""
    
    results = {"config_a": {}, "config_b": {}}
    
    for config_name, config in [("config_a", config_a), ("config_b", config_b)]:
        rag_system = RAGWithGuardrails(**config)
        
        metrics = {
            "success_rate": 0,
            "avg_fidelity": 0,
            "avg_response_time": 0,
            "hallucination_rate": 0
        }
        
        for query in test_queries:
            start_time = time.time()
            result = rag_system.query_with_guardrails(query)
            end_time = time.time()
            
            metrics["avg_response_time"] += (end_time - start_time)
            
            if result["status"] == "success":
                metrics["success_rate"] += 1
                metrics["avg_fidelity"] += result.get("fidelity_score", 0)
                
                if result.get("hallucination_detected", False):
                    metrics["hallucination_rate"] += 1
        
        # Normalizar
        total_queries = len(test_queries)
        metrics["success_rate"] /= total_queries
        metrics["avg_fidelity"] /= total_queries
        metrics["avg_response_time"] /= total_queries
        metrics["hallucination_rate"] /= total_queries
        
        results[config_name] = metrics
    
    # Comparação estatística
    comparison = compare_configs(results["config_a"], results["config_b"])
    results["statistical_significance"] = comparison
    
    return results
```

---

## 🚀 **Próximos Passos**

### **Implementações Futuras**

**1. Guardrails com Machine Learning:**
```python
# Modelo treinado para detectar alucinações
class HallucinationDetector:
    def __init__(self):
        self.model = load_pretrained_model("hallucination_detector.pkl")
    
    def predict(self, response: str, context: str) -> float:
        features = extract_features(response, context)
        return self.model.predict_proba([features])[0][1]  # Probabilidade de alucinação
```

**2. Guardrails Adaptativos:**
```python
# Sistema que aprende e ajusta thresholds automaticamente
class AdaptiveGuardrails:
    def __init__(self):
        self.feedback_history = []
        self.current_threshold = 0.35
    
    def update_threshold(self, user_feedback: float):
        """Ajusta threshold baseado no feedback do usuário."""
        self.feedback_history.append(user_feedback)
        
        if len(self.feedback_history) >= 10:
            avg_satisfaction = np.mean(self.feedback_history[-10:])
            
            if avg_satisfaction < 3.0:
                self.current_threshold -= 0.02  # Ser mais rigoroso
            elif avg_satisfaction > 4.0:
                self.current_threshold += 0.02  # Ser mais permissivo
```

**3. Integração com Ferramentas Externas:**
```python
# Integração com Guardrails AI
from guardrails import Guard
from guardrails.validators import ValidLength, RegexMatch

guard = Guard.from_rail_string("""
<rail version="0.1">
<output>
    <string name="response" 
            validators="valid-length: 10 500, regex-match: ^✅.*"
            on-fail="exception"/>
</output>
</rail>
""")

# Uso integrado
validated_response = guard(llm_function)(prompt)
```

### **Roadmap de Melhorias**

1. **Curto Prazo (1-2 semanas):**
   - Implementar threshold adaptativo por tipo de query
   - Adicionar mais validações de domínio específico
   - Melhorar algoritmos de detecção de alucinação

2. **Médio Prazo (1-2 meses):**
   - Treinar modelo ML para detecção de alucinações
   - Implementar A/B testing automático de configurações
   - Desenvolver dashboard de monitoramento avançado

3. **Longo Prazo (3+ meses):**
   - Integração com ferramentas externas (Guardrails AI, NeMo)
   - Sistema de feedback loop automático
   - Guardrails específicos por usuário/contexto

---

**Os Guardrails são a diferença entre um protótipo interessante e um sistema confiável para produção!** 🛡️

Implemente gradualmente, monitore constantemente, e ajuste baseado em dados reais. A segurança e qualidade do seu sistema RAG dependem dessas proteções.
