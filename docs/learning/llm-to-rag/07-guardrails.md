# Capítulo 07 — Guardrails: Controlando e Validando Respostas de IA

Sistemas RAG podem ser poderosos, mas também impredizíveis. Um LLM pode "alucinar", inventar informações ou sair do escopo desejado. **Guardrails** são mecanismos de controle que garantem que as respostas sejam seguras, precisas e adequadas ao contexto.

Neste capítulo, exploramos como implementar guardrails eficazes, desde validações simples até integrações com ferramentas especializadas como NeMo Guardrails e Guardrails AI.

**Resumo:** neste capítulo implementamos sistemas de guardrails para controlar respostas de IA, limitando-as ao contexto fornecido e integrando ferramentas externas de validação.

**Sumário:**
1. O que são guardrails e por que importam
2. Tipos de guardrails em sistemas RAG
3. Limitando respostas ao contexto
4. Guardrails de segurança e conformidade
5. Implementação prática com validadores
6. Integração com NeMo Guardrails
7. Integração com Guardrails AI
8. Monitoramento e métricas de guardrails
9. Conclusão

---

## 1. O que são guardrails e por que importam

### 1.1. Definição

**Guardrails** são sistemas de validação e controle que:
- Verificam se as respostas estão dentro do escopo esperado
- Bloqueiam conteúdo inadequado ou perigoso
- Garantem que o modelo siga políticas e diretrizes
- Monitoram qualidade e precisão das respostas

### 1.2. Por que são essenciais

```python
# Exemplo do problema sem guardrails
query = "Como fazer explosivos caseiros?"
response_sem_guardrail = llm.invoke(query)
# Resposta: "Para fazer explosivos, você precisa de..."

# Com guardrails
query = "Como fazer explosivos caseiros?"
response_com_guardrail = "Desculpe, não posso fornecer informações sobre criação de explosivos ou substâncias perigosas."
```

**Riscos sem guardrails:**
- Informações perigosas ou ilegais
- Alucinações apresentadas como fatos
- Respostas fora do domínio de conhecimento
- Violação de políticas empresariais
- Experiência inconsistente do usuário

### 1.3. Tipos de problemas que guardrails resolvem

- **Alucinação:** modelo inventa informações não presentes no contexto
- **Scope creep:** resposta foge do domínio esperado
- **Unsafe content:** conteúdo perigoso, tóxico ou inadequado
- **Privacy leaks:** exposição acidental de informações sensíveis
- **Inconsistência:** respostas contraditórias para perguntas similares

---

## 2. Tipos de guardrails em sistemas RAG

### 2.1. Guardrails de entrada (Input Guardrails)

Validam consultas antes do processamento:

```python
class InputGuardrails:
    def __init__(self):
        self.blocked_topics = [
            "explosivos", "drogas", "hacking", "violência",
            "informações pessoais", "senhas", "dados bancários"
        ]
        
        self.sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
        ]
    
    def validate_input(self, query: str) -> dict:
        """Valida se a consulta é segura e apropriada"""
        result = {
            "is_safe": True,
            "blocked_reason": None,
            "sanitized_query": query,
            "warnings": []
        }
        
        query_lower = query.lower()
        
        # Verifica tópicos bloqueados
        for topic in self.blocked_topics:
            if topic in query_lower:
                result["is_safe"] = False
                result["blocked_reason"] = f"Tópico não permitido: {topic}"
                return result
        
        # Verifica padrões sensíveis
        for pattern in self.sensitive_patterns:
            if re.search(pattern, query):
                result["warnings"].append("Possível informação sensível detectada")
                # Sanitiza removendo o padrão
                result["sanitized_query"] = re.sub(pattern, "[DADOS_REMOVIDOS]", query)
        
        return result
```

### 2.2. Guardrails de contexto (Context Guardrails)

Validam se o contexto recuperado é apropriado:

```python
class ContextGuardrails:
    def __init__(self):
        self.min_relevance_threshold = 0.3
        self.max_context_length = 4000
        self.blocked_sources = ["confidencial", "interno", "draft"]
    
    def validate_context(self, chunks: List, query: str) -> dict:
        """Valida se o contexto é adequado para resposta"""
        result = {
            "is_valid": True,
            "filtered_chunks": [],
            "issues": [],
            "context_quality": 0.0
        }
        
        if not chunks:
            result["is_valid"] = False
            result["issues"].append("Nenhum contexto encontrado")
            return result
        
        total_relevance = 0
        valid_chunks = []
        total_length = 0
        
        for chunk in chunks:
            # Verifica fonte bloqueada
            source = chunk.metadata.get("source", "").lower()
            if any(blocked in source for blocked in self.blocked_sources):
                result["issues"].append(f"Fonte bloqueada: {source}")
                continue
            
            # Calcula relevância
            relevance = self._calculate_relevance(chunk.page_content, query)
            
            if relevance >= self.min_relevance_threshold:
                if total_length + len(chunk.page_content) <= self.max_context_length:
                    valid_chunks.append(chunk)
                    total_relevance += relevance
                    total_length += len(chunk.page_content)
                else:
                    break
        
        if not valid_chunks:
            result["is_valid"] = False
            result["issues"].append("Nenhum contexto relevante encontrado")
        else:
            result["filtered_chunks"] = valid_chunks
            result["context_quality"] = total_relevance / len(valid_chunks)
        
        return result
    
    def _calculate_relevance(self, content: str, query: str) -> float:
        """Calcula relevância simples baseada em sobreposição de palavras"""
        content_words = set(content.lower().split())
        query_words = set(query.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(content_words.intersection(query_words))
        return overlap / len(query_words)
```

### 2.3. Guardrails de saída (Output Guardrails)

Validam respostas antes de entregá-las ao usuário:

```python
class OutputGuardrails:
    def __init__(self):
        self.forbidden_phrases = [
            "não sei", "não posso ajudar", "não tenho informação",
            "desculpe, mas", "infelizmente não"
        ]
        
        self.quality_indicators = [
            "baseado no contexto", "de acordo com", "segundo o documento",
            "conforme mencionado", "como indicado"
        ]
        
        self.unsafe_content_patterns = [
            r'\b(hack|crack|exploit)\b',
            r'\b(bomb|explosive|weapon)\b',
            r'\b(kill|murder|harm)\b'
        ]
    
    def validate_output(self, response: str, context_chunks: List, query: str) -> dict:
        """Valida se a resposta é segura e apropriada"""
        result = {
            "is_safe": True,
            "is_grounded": True,
            "quality_score": 0.0,
            "issues": [],
            "enhanced_response": response
        }
        
        response_lower = response.lower()
        
        # 1. Verifica conteúdo inseguro
        for pattern in self.unsafe_content_patterns:
            if re.search(pattern, response_lower, re.IGNORECASE):
                result["is_safe"] = False
                result["issues"].append("Conteúdo potencialmente inseguro detectado")
                return result
        
        # 2. Verifica se está fundamentada no contexto
        grounding_score = self._calculate_grounding(response, context_chunks)
        if grounding_score < 0.3:
            result["is_grounded"] = False
            result["issues"].append("Resposta não fundamentada no contexto")
        
        # 3. Verifica qualidade da resposta
        quality_score = self._calculate_quality(response, query)
        result["quality_score"] = quality_score
        
        if quality_score < 0.5:
            result["issues"].append("Resposta de baixa qualidade")
        
        # 4. Adiciona disclaimers se necessário
        if result["is_safe"] and result["is_grounded"]:
            result["enhanced_response"] = self._add_disclaimers(response, grounding_score)
        
        return result
    
    def _calculate_grounding(self, response: str, context_chunks: List) -> float:
        """Calcula quanto da resposta está fundamentada no contexto"""
        if not context_chunks:
            return 0.0
        
        response_words = set(response.lower().split())
        context_words = set()
        
        for chunk in context_chunks:
            context_words.update(chunk.page_content.lower().split())
        
        if not response_words:
            return 0.0
        
        grounded_words = response_words.intersection(context_words)
        return len(grounded_words) / len(response_words)
    
    def _calculate_quality(self, response: str, query: str) -> float:
        """Calcula score de qualidade da resposta"""
        score = 0.5  # Base score
        
        # Bonus por indicadores de qualidade
        response_lower = response.lower()
        for indicator in self.quality_indicators:
            if indicator in response_lower:
                score += 0.1
        
        # Penalidade por frases proibidas
        for phrase in self.forbidden_phrases:
            if phrase in response_lower:
                score -= 0.3
        
        # Penalidade por resposta muito curta
        if len(response) < 50:
            score -= 0.2
        
        # Bonus por resposta bem estruturada
        if '.' in response and len(response.split('.')) > 1:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _add_disclaimers(self, response: str, grounding_score: float) -> str:
        """Adiciona disclaimers apropriados à resposta"""
        if grounding_score < 0.7:
            disclaimer = "\n\n*Nota: Esta resposta é baseada nas informações disponíveis e pode não estar completa.*"
            return response + disclaimer
        
        return response
```

---

## 3. Limitando respostas ao contexto

### 3.1. Sistema de validação estrita

```python
class ContextOnlyGuardrail:
    def __init__(self, strictness_level="medium"):
        self.strictness_level = strictness_level
        self.thresholds = {
            "strict": {"min_grounding": 0.8, "max_extrapolation": 0.1},
            "medium": {"min_grounding": 0.6, "max_extrapolation": 0.3},
            "lenient": {"min_grounding": 0.4, "max_extrapolation": 0.5}
        }
    
    def enforce_context_only(self, response: str, context_chunks: List, query: str) -> dict:
        """Força resposta a ser baseada apenas no contexto"""
        threshold = self.thresholds[self.strictness_level]
        
        # Analisa fundamentação
        grounding_analysis = self._analyze_grounding(response, context_chunks)
        
        if grounding_analysis["grounding_score"] < threshold["min_grounding"]:
            # Resposta não suficientemente fundamentada
            return {
                "approved": False,
                "reason": "Resposta não fundamentada no contexto fornecido",
                "suggested_response": self._generate_context_only_response(context_chunks, query),
                "grounding_score": grounding_analysis["grounding_score"]
            }
        
        if grounding_analysis["extrapolation_score"] > threshold["max_extrapolation"]:
            # Muita extrapolação
            return {
                "approved": False,
                "reason": "Resposta contém muita informação não presente no contexto",
                "suggested_response": self._filter_to_context_only(response, context_chunks),
                "extrapolation_score": grounding_analysis["extrapolation_score"]
            }
        
        return {
            "approved": True,
            "enhanced_response": self._add_source_citations(response, context_chunks),
            "grounding_score": grounding_analysis["grounding_score"]
        }
    
    def _analyze_grounding(self, response: str, context_chunks: List) -> dict:
        """Analisa quanto da resposta está fundamentada no contexto"""
        if not context_chunks:
            return {"grounding_score": 0.0, "extrapolation_score": 1.0}
        
        # Extrai frases da resposta
        response_sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        grounded_sentences = 0
        total_sentences = len(response_sentences)
        
        # Combina todo o contexto
        full_context = " ".join([chunk.page_content for chunk in context_chunks]).lower()
        
        for sentence in response_sentences:
            sentence_lower = sentence.lower()
            # Verifica se a essência da frase está no contexto
            sentence_words = set(sentence_lower.split())
            
            # Remove palavras muito comuns
            common_words = {"o", "a", "os", "as", "de", "da", "do", "em", "para", "com", "por", "é", "são", "foi", "um", "uma"}
            meaningful_words = sentence_words - common_words
            
            if meaningful_words:
                # Verifica sobreposição significativa
                context_words = set(full_context.split())
                overlap = len(meaningful_words.intersection(context_words))
                
                if overlap / len(meaningful_words) >= 0.5:  # 50% das palavras significativas
                    grounded_sentences += 1
        
        grounding_score = grounded_sentences / total_sentences if total_sentences > 0 else 0
        extrapolation_score = 1.0 - grounding_score
        
        return {
            "grounding_score": grounding_score,
            "extrapolation_score": extrapolation_score,
            "grounded_sentences": grounded_sentences,
            "total_sentences": total_sentences
        }
    
    def _generate_context_only_response(self, context_chunks: List, query: str) -> str:
        """Gera resposta baseada estritamente no contexto"""
        if not context_chunks:
            return "Não encontrei informações suficientes no contexto fornecido para responder sua pergunta."
        
        # Extrai informações mais relevantes do contexto
        relevant_info = []
        query_words = set(query.lower().split())
        
        for chunk in context_chunks:
            content = chunk.page_content
            content_words = set(content.lower().split())
            
            # Calcula relevância
            overlap = len(query_words.intersection(content_words))
            if overlap > 0:
                relevant_info.append(content)
        
        if relevant_info:
            # Combina informações relevantes
            combined_info = " ".join(relevant_info[:2])  # Máximo 2 chunks
            return f"Baseado no contexto fornecido: {combined_info}"
        else:
            return "O contexto fornecido não contém informações suficientes para responder sua pergunta."
    
    def _add_source_citations(self, response: str, context_chunks: List) -> str:
        """Adiciona citações das fontes à resposta"""
        if not context_chunks:
            return response
        
        sources = []
        for chunk in context_chunks:
            source = chunk.metadata.get("source", "documento")
            if source not in sources:
                sources.append(source)
        
        if sources:
            citation = f"\n\nFontes: {', '.join(sources)}"
            return response + citation
        
        return response
```

### 3.2. Template de prompt restritivo

```python
class RestrictivePromptTemplate:
    def __init__(self):
        self.template = """
Você é um assistente que responde EXCLUSIVAMENTE baseado no contexto fornecido.

REGRAS OBRIGATÓRIAS:
1. Use APENAS informações presentes no contexto
2. Se a informação não estiver no contexto, diga claramente
3. Não adicione conhecimento externo
4. Não faça suposições ou extrapolações
5. Cite a fonte quando possível

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA (seguindo rigorosamente as regras acima):
"""
    
    def create_restricted_prompt(self, context: str, question: str) -> str:
        """Cria prompt que força aderência ao contexto"""
        return self.template.format(context=context, question=question)
    
    def validate_response_adherence(self, response: str, context: str) -> bool:
        """Valida se a resposta aderiu às restrições"""
        
        # Verifica se há frases indicativas de não-aderência
        non_adherent_phrases = [
            "baseado no meu conhecimento",
            "geralmente",
            "normalmente",
            "tipicamente",
            "em geral",
            "segundo a literatura",
            "é conhecido que"
        ]
        
        response_lower = response.lower()
        for phrase in non_adherent_phrases:
            if phrase in response_lower:
                return False
        
        # Verifica se há disclaimer apropriado quando não sabe
        if "não" in response_lower and any(word in response_lower for word in ["encontrei", "contexto", "informação"]):
            return True  # Apropriadamente admitiu falta de informação
        
        return True
```

---

## 4. Guardrails de segurança e conformidade

### 4.1. Detector de conteúdo tóxico

```python
import re
from typing import Dict, List

class ToxicityDetector:
    def __init__(self):
        # Padrões de conteúdo problemático
        self.toxicity_patterns = {
            "hate_speech": [
                r'\b(idiota|burro|estúpido)\b',
                r'\b(ódio|raiva|violência)\b'
            ],
            "personal_attacks": [
                r'\bvocê é\s+(idiota|burro|incompetente)\b',
                r'\b(seu|sua)\s+(idiota|burro)\b'
            ],
            "discriminatory": [
                r'\b(inferior|superior)\s+por\s+(cor|raça|gênero)\b'
            ],
            "violent": [
                r'\b(matar|assassinar|violentar)\b',
                r'\b(bomba|explosivo|arma)\b'
            ]
        }
        
        self.severity_weights = {
            "hate_speech": 0.8,
            "personal_attacks": 0.9,
            "discriminatory": 1.0,
            "violent": 1.0
        }
    
    def detect_toxicity(self, text: str) -> Dict:
        """Detecta e classifica conteúdo tóxico"""
        results = {
            "is_toxic": False,
            "toxicity_score": 0.0,
            "detected_categories": [],
            "flagged_phrases": []
        }
        
        text_lower = text.lower()
        max_severity = 0.0
        
        for category, patterns in self.toxicity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    results["detected_categories"].append(category)
                    results["flagged_phrases"].extend(matches)
                    
                    severity = self.severity_weights[category]
                    max_severity = max(max_severity, severity)
        
        if max_severity > 0:
            results["is_toxic"] = True
            results["toxicity_score"] = max_severity
        
        return results
    
    def sanitize_text(self, text: str) -> str:
        """Remove ou substitui conteúdo tóxico"""
        sanitized = text
        
        for category, patterns in self.toxicity_patterns.items():
            for pattern in patterns:
                sanitized = re.sub(pattern, "[CONTEÚDO_REMOVIDO]", sanitized, flags=re.IGNORECASE)
        
        return sanitized
```

### 4.2. Validador de conformidade

```python
class ComplianceValidator:
    def __init__(self, industry="general"):
        self.industry = industry
        self.compliance_rules = self._load_compliance_rules()
    
    def _load_compliance_rules(self) -> Dict:
        """Carrega regras de conformidade por setor"""
        rules = {
            "general": {
                "forbidden_topics": ["informações pessoais", "dados bancários"],
                "required_disclaimers": ["Esta informação é apenas educativa"],
                "max_response_length": 2000
            },
            "healthcare": {
                "forbidden_topics": ["diagnóstico médico", "prescrição", "tratamento"],
                "required_disclaimers": ["Consulte um profissional de saúde"],
                "max_response_length": 1500
            },
            "financial": {
                "forbidden_topics": ["conselhos de investimento", "dados bancários"],
                "required_disclaimers": ["Esta não é uma recomendação financeira"],
                "max_response_length": 1200
            }
        }
        
        return rules.get(self.industry, rules["general"])
    
    def validate_compliance(self, response: str, query: str) -> Dict:
        """Valida conformidade da resposta com regulamentações"""
        result = {
            "is_compliant": True,
            "violations": [],
            "required_actions": [],
            "enhanced_response": response
        }
        
        # Verifica tópicos proibidos
        for topic in self.compliance_rules["forbidden_topics"]:
            if topic.lower() in response.lower() or topic.lower() in query.lower():
                result["is_compliant"] = False
                result["violations"].append(f"Tópico proibido: {topic}")
                result["required_actions"].append(f"Remover referências a {topic}")
        
        # Verifica tamanho da resposta
        if len(response) > self.compliance_rules["max_response_length"]:
            result["violations"].append("Resposta muito longa")
            result["required_actions"].append("Encurtar resposta")
        
        # Adiciona disclaimers obrigatórios
        if result["is_compliant"]:
            enhanced_response = response
            for disclaimer in self.compliance_rules["required_disclaimers"]:
                if disclaimer not in response:
                    enhanced_response += f"\n\n*{disclaimer}*"
            
            result["enhanced_response"] = enhanced_response
        
        return result
```

---

## 5. Implementação prática com validadores

### 5.1. Sistema de guardrails integrado

```python
class IntegratedGuardrailSystem:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Inicializa validadores
        self.input_guardrails = InputGuardrails()
        self.context_guardrails = ContextGuardrails()
        self.output_guardrails = OutputGuardrails()
        self.context_only_guardrail = ContextOnlyGuardrail(
            strictness_level=self.config.get("strictness", "medium")
        )
        self.toxicity_detector = ToxicityDetector()
        self.compliance_validator = ComplianceValidator(
            industry=self.config.get("industry", "general")
        )
        
        # Métricas
        self.blocked_queries = 0
        self.blocked_responses = 0
        self.total_queries = 0
    
    def process_query(self, query: str, retriever, llm) -> Dict:
        """Processa consulta com todos os guardrails"""
        self.total_queries += 1
        
        # 1. Validação de entrada
        input_validation = self.input_guardrails.validate_input(query)
        if not input_validation["is_safe"]:
            self.blocked_queries += 1
            return {
                "success": False,
                "reason": "Consulta bloqueada",
                "details": input_validation["blocked_reason"],
                "response": "Desculpe, não posso processar essa consulta."
            }
        
        sanitized_query = input_validation["sanitized_query"]
        
        # 2. Recuperação de contexto
        try:
            retrieved_chunks = retriever.get_relevant_documents(sanitized_query)
        except Exception as e:
            return {
                "success": False,
                "reason": "Erro na recuperação",
                "details": str(e),
                "response": "Erro interno. Tente novamente."
            }
        
        # 3. Validação de contexto
        context_validation = self.context_guardrails.validate_context(retrieved_chunks, sanitized_query)
        if not context_validation["is_valid"]:
            return {
                "success": False,
                "reason": "Contexto inadequado",
                "details": context_validation["issues"],
                "response": "Não encontrei informações adequadas para responder sua pergunta."
            }
        
        valid_chunks = context_validation["filtered_chunks"]
        
        # 4. Geração de resposta
        context_text = "\n\n".join([chunk.page_content for chunk in valid_chunks])
        
        prompt_template = RestrictivePromptTemplate()
        prompt = prompt_template.create_restricted_prompt(context_text, sanitized_query)
        
        try:
            response = llm.invoke(prompt).content
        except Exception as e:
            return {
                "success": False,
                "reason": "Erro na geração",
                "details": str(e),
                "response": "Erro interno. Tente novamente."
            }
        
        # 5. Validação de saída
        output_validation = self.output_guardrails.validate_output(response, valid_chunks, sanitized_query)
        if not output_validation["is_safe"]:
            self.blocked_responses += 1
            return {
                "success": False,
                "reason": "Resposta bloqueada",
                "details": output_validation["issues"],
                "response": "Não posso fornecer uma resposta adequada para essa pergunta."
            }
        
        # 6. Validação de aderência ao contexto
        context_validation = self.context_only_guardrail.enforce_context_only(response, valid_chunks, sanitized_query)
        if not context_validation["approved"]:
            response = context_validation["suggested_response"]
        else:
            response = context_validation.get("enhanced_response", response)
        
        # 7. Detecção de toxicidade
        toxicity_check = self.toxicity_detector.detect_toxicity(response)
        if toxicity_check["is_toxic"]:
            self.blocked_responses += 1
            return {
                "success": False,
                "reason": "Conteúdo tóxico detectado",
                "details": toxicity_check["detected_categories"],
                "response": "Não posso fornecer essa resposta."
            }
        
        # 8. Validação de conformidade
        compliance_check = self.compliance_validator.validate_compliance(response, sanitized_query)
        if not compliance_check["is_compliant"]:
            return {
                "success": False,
                "reason": "Violação de conformidade",
                "details": compliance_check["violations"],
                "response": "Não posso fornecer essa informação devido a restrições de conformidade."
            }
        
        final_response = compliance_check["enhanced_response"]
        
        return {
            "success": True,
            "response": final_response,
            "metadata": {
                "chunks_used": len(valid_chunks),
                "sources": [chunk.metadata.get("source") for chunk in valid_chunks],
                "grounding_score": context_validation.get("grounding_score", 0),
                "quality_score": output_validation["quality_score"],
                "compliance_status": "approved"
            }
        }
    
    def get_guardrail_stats(self) -> Dict:
        """Retorna estatísticas dos guardrails"""
        return {
            "total_queries": self.total_queries,
            "blocked_queries": self.blocked_queries,
            "blocked_responses": self.blocked_responses,
            "block_rate_queries": self.blocked_queries / self.total_queries if self.total_queries > 0 else 0,
            "block_rate_responses": self.blocked_responses / self.total_queries if self.total_queries > 0 else 0
        }
```

---

## 6. Integração com NeMo Guardrails

### 6.1. Configuração do NeMo Guardrails

```python
# Instalação: pip install nemoguardrails

from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.actions.actions import ActionResult

class NeMoGuardrailsIntegration:
    def __init__(self, config_path: str = None):
        # Configuração via código (alternativa ao arquivo YAML)
        config = RailsConfig.from_content(
            colang_content=self._get_colang_config(),
            yaml_content=self._get_yaml_config()
        )
        
        self.rails = LLMRails(config)
    
    def _get_colang_config(self) -> str:
        """Configuração Colang para definir fluxos e restrições"""
        return """
# Definir intenções do usuário
define user ask_about_harmful_content
  "como fazer explosivos"
  "como hackear"
  "informações pessoais"

define user ask_about_outside_scope
  "qual é a capital da França"
  "como cozinhar"
  "previsão do tempo"

# Definir fluxos de bot
define bot refuse_harmful_request
  "Desculpe, não posso fornecer informações sobre esse tópico."

define bot redirect_to_scope
  "Posso ajudar apenas com informações relacionadas aos documentos fornecidos."

# Fluxos principais
define flow harmful_content_handling
  user ask_about_harmful_content
  bot refuse_harmful_request

define flow scope_enforcement
  user ask_about_outside_scope
  bot redirect_to_scope

# Guardrails de entrada
define flow input_rail
  user ...
  if $user_message.contains_harmful_content
    bot refuse_harmful_request
    stop

# Guardrails de saída
define flow output_rail
  bot ...
  if $bot_message.contains_personal_info
    bot "Não posso compartilhar informações pessoais."
    stop
"""
    
    def _get_yaml_config(self) -> str:
        """Configuração YAML para modelos e parâmetros"""
        return """
models:
  - type: main
    engine: ollama
    model: llama3

rails:
  input:
    flows:
      - harmful_content_detection
      - scope_validation
  
  output:
    flows:
      - personal_info_filter
      - context_grounding_check

instructions:
  - type: general
    content: |
      Você deve responder apenas baseado no contexto fornecido.
      Não invente informações.
      Se não souber, diga claramente.
      
prompts:
  - task: general
    content: |
      Baseado exclusivamente no contexto abaixo, responda a pergunta:
      
      Contexto: {context}
      Pergunta: {user_message}
      
      Resposta:
"""
    
    async def process_with_nemo(self, query: str, context: str) -> Dict:
        """Processa consulta usando NeMo Guardrails"""
        try:
            # Adiciona contexto à sessão
            response = await self.rails.generate_async(
                messages=[{
                    "role": "user", 
                    "content": query
                }],
                context={
                    "context": context
                }
            )
            
            return {
                "success": True,
                "response": response["content"],
                "rails_info": response.get("rails_info", {})
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": "Erro no processamento com guardrails."
            }
```

### 6.2. Actions customizadas para NeMo

```python
from nemoguardrails.actions import action

@action
async def check_context_grounding(context: str, bot_response: str) -> ActionResult:
    """Action customizada para verificar fundamentação no contexto"""
    
    # Análise de fundamentação
    context_words = set(context.lower().split())
    response_words = set(bot_response.lower().split())
    
    # Remove palavras comuns
    common_words = {"o", "a", "os", "as", "de", "da", "do", "em", "para", "com", "por"}
    meaningful_response_words = response_words - common_words
    
    if not meaningful_response_words:
        return ActionResult(
            return_value={"grounded": False, "score": 0.0}
        )
    
    # Calcula sobreposição
    overlap = len(meaningful_response_words.intersection(context_words))
    grounding_score = overlap / len(meaningful_response_words)
    
    is_grounded = grounding_score >= 0.3  # Threshold configurável
    
    return ActionResult(
        return_value={
            "grounded": is_grounded,
            "score": grounding_score,
            "threshold": 0.3
        }
    )

@action  
async def validate_response_quality(bot_response: str) -> ActionResult:
    """Action para validar qualidade da resposta"""
    
    quality_score = 0.5  # Base
    
    # Verifica tamanho adequado
    if 50 <= len(bot_response) <= 1000:
        quality_score += 0.2
    
    # Verifica estrutura
    if '.' in bot_response and len(bot_response.split('.')) > 1:
        quality_score += 0.2
    
    # Verifica indicadores de qualidade
    quality_indicators = ["baseado", "segundo", "conforme", "de acordo"]
    if any(indicator in bot_response.lower() for indicator in quality_indicators):
        quality_score += 0.1
    
    is_quality = quality_score >= 0.6
    
    return ActionResult(
        return_value={
            "is_quality": is_quality,
            "score": quality_score
        }
    )
```

---

## 7. Integração com Guardrails AI

### 7.1. Configuração do Guardrails AI

```python
# Instalação: pip install guardrails-ai

import guardrails as gd
from guardrails.validators import ValidLength, ToxicLanguage, ReadingTime
from guardrails import Guard

class GuardrailsAIIntegration:
    def __init__(self):
        # Define guard com validadores
        self.guard = Guard().use_many(
            ValidLength(min=20, max=2000, on_fail="reask"),
            ToxicLanguage(threshold=0.8, validation_method="sentence", on_fail="filter"),
            ReadingTime(reading_level=8, max_time=120, on_fail="reask")
        )
        
        # Guard específico para fundamentação no contexto
        self.context_guard = self._create_context_guard()
    
    def _create_context_guard(self):
        """Cria guard customizado para verificar fundamentação no contexto"""
        
        @gd.validator(name="context-grounding", data_type="string")
        def validate_context_grounding(value: str, metadata: dict) -> str:
            """Valida se a resposta está fundamentada no contexto"""
            context = metadata.get("context", "")
            
            if not context:
                raise gd.ValidationError("Contexto não fornecido")
            
            # Análise de fundamentação
            value_words = set(value.lower().split())
            context_words = set(context.lower().split())
            
            # Remove palavras comuns
            common_words = {"o", "a", "os", "as", "de", "da", "do", "em", "para", "com", "por", "é", "são"}
            meaningful_words = value_words - common_words
            
            if not meaningful_words:
                raise gd.ValidationError("Resposta sem conteúdo significativo")
            
            # Calcula fundamentação
            grounded_words = meaningful_words.intersection(context_words)
            grounding_score = len(grounded_words) / len(meaningful_words)
            
            if grounding_score < 0.3:
                raise gd.ValidationError(
                    f"Resposta insuficientemente fundamentada no contexto (score: {grounding_score:.2f})"
                )
            
            return value
        
        return Guard().use(validate_context_grounding, on_fail="reask")
    
    def validate_response(self, response: str, context: str = "", query: str = "") -> Dict:
        """Valida resposta usando Guardrails AI"""
        
        try:
            # Validação geral
            general_result = self.guard.validate(
                response,
                metadata={
                    "context": context,
                    "query": query
                }
            )
            
            # Validação de fundamentação no contexto
            if context:
                context_result = self.context_guard.validate(
                    response,
                    metadata={"context": context}
                )
                
                return {
                    "success": True,
                    "validated_response": context_result.validated_output,
                    "validation_passed": context_result.validation_passed,
                    "errors": context_result.error or []
                }
            else:
                return {
                    "success": True,
                    "validated_response": general_result.validated_output,
                    "validation_passed": general_result.validation_passed,
                    "errors": general_result.error or []
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "validated_response": None
            }
    
    def create_schema_validator(self) -> Guard:
        """Cria validador baseado em schema estruturado"""
        from pydantic import BaseModel, Field
        
        class ValidatedResponse(BaseModel):
            answer: str = Field(
                description="Resposta baseada exclusivamente no contexto fornecido",
                validators=[
                    ValidLength(min=20, max=1500),
                    ToxicLanguage(threshold=0.9)
                ]
            )
            confidence: float = Field(
                description="Nível de confiança na resposta (0-1)",
                ge=0.0,
                le=1.0
            )
            sources: list = Field(
                description="Lista das fontes utilizadas",
                default_factory=list
            )
            grounding_score: float = Field(
                description="Score de fundamentação no contexto (0-1)",
                ge=0.0,
                le=1.0
            )
        
        return Guard.from_pydantic(ValidatedResponse)
```

### 7.2. Validadores customizados

```python
from guardrails.validator_base import Validator, ValidationResult

class ContextGroundingValidator(Validator):
    """Validador customizado para fundamentação no contexto"""
    
    def __init__(self, threshold: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
    
    def validate(self, value: str, metadata: dict) -> ValidationResult:
        """Valida fundamentação da resposta no contexto"""
        context = metadata.get("context", "")
        
        if not context:
            return ValidationResult(
                outcome="fail",
                error_message="Contexto não fornecido para validação"
            )
        
        grounding_score = self._calculate_grounding(value, context)
        
        if grounding_score < self.threshold:
            return ValidationResult(
                outcome="fail",
                error_message=f"Resposta insuficientemente fundamentada (score: {grounding_score:.2f}, threshold: {self.threshold})",
                fix_value=self._suggest_grounded_response(context)
            )
        
        return ValidationResult(
            outcome="pass",
            metadata={"grounding_score": grounding_score}
        )
    
    def _calculate_grounding(self, response: str, context: str) -> float:
        """Calcula score de fundamentação"""
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())
        
        # Remove palavras muito comuns
        common_words = {"o", "a", "os", "as", "de", "da", "do", "em", "para", "com", "por", "é", "são", "foi", "um", "uma"}
        meaningful_words = response_words - common_words
        
        if not meaningful_words:
            return 0.0
        
        grounded_words = meaningful_words.intersection(context_words)
        return len(grounded_words) / len(meaningful_words)
    
    def _suggest_grounded_response(self, context: str) -> str:
        """Sugere resposta mais fundamentada no contexto"""
        # Simplificado - extrai primeira frase relevante do contexto
        sentences = context.split('.')
        if sentences:
            return f"Baseado no contexto: {sentences[0].strip()}."
        return "Baseado no contexto fornecido, não posso elaborar uma resposta adequada."

class ScopeValidator(Validator):
    """Validador para manter respostas no escopo"""
    
    def __init__(self, allowed_topics: list = None, **kwargs):
        super().__init__(**kwargs)
        self.allowed_topics = allowed_topics or []
    
    def validate(self, value: str, metadata: dict) -> ValidationResult:
        """Valida se a resposta está no escopo permitido"""
        
        if not self.allowed_topics:
            return ValidationResult(outcome="pass")
        
        value_lower = value.lower()
        
        # Verifica se contém tópicos permitidos
        contains_allowed_topic = any(
            topic.lower() in value_lower 
            for topic in self.allowed_topics
        )
        
        if not contains_allowed_topic:
            return ValidationResult(
                outcome="fail",
                error_message="Resposta fora do escopo permitido",
                fix_value="Posso ajudar apenas com tópicos relacionados aos documentos fornecidos."
            )
        
        return ValidationResult(outcome="pass")
```

---

## 8. Monitoramento e métricas de guardrails

### 8.1. Dashboard de guardrails

```python
import json
from datetime import datetime, timedelta

class GuardrailsMonitor:
    def __init__(self):
        self.events = []
        self.metrics = {
            "total_queries": 0,
            "blocked_input": 0,
            "blocked_output": 0,
            "context_failures": 0,
            "toxicity_blocks": 0,
            "compliance_violations": 0
        }
    
    def log_event(self, event_type: str, details: dict):
        """Registra evento de guardrail"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "details": details
        }
        
        self.events.append(event)
        self.metrics["total_queries"] += 1
        
        # Atualiza métricas específicas
        if event_type == "input_blocked":
            self.metrics["blocked_input"] += 1
        elif event_type == "output_blocked":
            self.metrics["blocked_output"] += 1
        elif event_type == "context_failure":
            self.metrics["context_failures"] += 1
        elif event_type == "toxicity_detected":
            self.metrics["toxicity_blocks"] += 1
        elif event_type == "compliance_violation":
            self.metrics["compliance_violations"] += 1
    
    def get_dashboard_data(self, hours: int = 24) -> dict:
        """Gera dados para dashboard de monitoramento"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent_events = [
            event for event in self.events
            if datetime.fromisoformat(event["timestamp"]) >= cutoff
        ]
        
        # Calcula métricas do período
        period_metrics = {
            "total_queries": len(recent_events),
            "block_rate": 0.0,
            "top_block_reasons": {},
            "hourly_distribution": {}
        }
        
        blocked_events = [
            event for event in recent_events
            if "blocked" in event["type"] or "violation" in event["type"] or "failure" in event["type"]
        ]
        
        if recent_events:
            period_metrics["block_rate"] = len(blocked_events) / len(recent_events)
        
        # Top razões de bloqueio
        for event in blocked_events:
            reason = event["details"].get("reason", "unknown")
            period_metrics["top_block_reasons"][reason] = period_metrics["top_block_reasons"].get(reason, 0) + 1
        
        # Distribuição por hora
        for event in recent_events:
            hour = datetime.fromisoformat(event["timestamp"]).hour
            period_metrics["hourly_distribution"][hour] = period_metrics["hourly_distribution"].get(hour, 0) + 1
        
        return period_metrics
    
    def generate_report(self) -> str:
        """Gera relatório textual dos guardrails"""
        dashboard_data = self.get_dashboard_data()
        
        report = f"""
=== RELATÓRIO DE GUARDRAILS ===
Período: Últimas 24 horas

📊 MÉTRICAS GERAIS:
- Total de consultas: {dashboard_data['total_queries']}
- Taxa de bloqueio: {dashboard_data['block_rate']:.2%}

🚫 PRINCIPAIS RAZÕES DE BLOQUEIO:
"""
        
        for reason, count in sorted(dashboard_data['top_block_reasons'].items(), key=lambda x: x[1], reverse=True):
            report += f"- {reason}: {count} vezes\n"
        
        report += f"""
⏰ DISTRIBUIÇÃO POR HORA:
"""
        
        for hour in sorted(dashboard_data['hourly_distribution'].keys()):
            count = dashboard_data['hourly_distribution'][hour]
            report += f"- {hour:02d}h: {count} consultas\n"
        
        return report
    
    def export_events(self, filepath: str):
        """Exporta eventos para análise"""
        with open(filepath, 'w') as f:
            json.dump(self.events, f, indent=2, ensure_ascii=False)
```

---

## 9. Conclusão

Implementamos um sistema completo de guardrails que:

- **Protege entradas:** valida consultas antes do processamento
- **Controla contexto:** garante que apenas informações relevantes sejam usadas
- **Limita respostas:** força aderência estrita ao contexto fornecido
- **Detecta toxicidade:** bloqueia conteúdo inadequado
- **Garante conformidade:** atende regulamentações setoriais
- **Monitora operação:** acompanha métricas e tendências

**Ferramentas integradas:**
- **NeMo Guardrails:** fluxos declarativos e controle de conversação
- **Guardrails AI:** validadores estruturados e schema-based
- **Sistema próprio:** validações customizadas e métricas

O sistema agora é **robusto, seguro e auditável**, pronto para ambientes de produção que exigem alto controle de qualidade e conformidade.

**Exercício prático:** configure os guardrails no seu projeto e teste com consultas problemáticas para verificar as proteções em ação.

---

### Pergunta ao leitor

Agora que temos guardrails robustos implementados, quer que eu escreva o **Capítulo 08 — Avaliação e Métricas de Qualidade** focando em como medir e melhorar continuamente a performance do sistema RAG?
