# Prompt Cheatsheet — Templates e Exemplos Rápidos

Este arquivo contém templates e exemplos prontos para copiar/colar em experimentos.

## Templates básicos

Zero‑shot
```
Explique em uma frase o que é <topico>.
```

Few‑shot (2 exemplares)
```
Exemplo 1:
Q: O que é cache?
A: Cache é um armazenamento temporário que acelera acesso a dados frequentemente usados.

Exemplo 2:
Q: O que é paralelismo?
A: Paralelismo é a execução simultânea de múltiplas tarefas para aumentar eficiência.

Agora responda:
Q: O que é <topico>?
A:
```

Instruction + Role
```
Você é um especialista em segurança. Responda de forma concisa (máx. 3 frases): <pergunta>
```

Chain‑of‑Thought (simplificado)
```
Pergunta: <pergunta>
Pense passo a passo e mostre o raciocínio antes da resposta.
Resposta:
```

RAG template
```
Você é um assistente que responde APENAS com base no contexto abaixo.

CONTEXT:
{context}

PERGUNTA: {question}

Se não estiver no contexto, responda: "Não encontrei essa informação no contexto fornecido." Cite as fontes.
```

Schema / JSON output
```
Extraia em JSON com campos: title, date(YYYY-MM-DD), summary.
Texto: <texto>
JSON:
```

Self‑consistency quick recipe
- Gerar N=5 respostas com seeds/temperatures diferentes.
- Escolher resposta mais frequente ou a que tiver maior score agregado.

---

Uso prático
- Cole o template no seu prompt builder.  
- Conte tokens antes de enviar (tiktoken).  
- Ajuste temperatura e top_p conforme objetivo (0–0.2 factual; 0.7+ criativo).

---

Fim.
