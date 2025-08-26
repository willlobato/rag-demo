# Capítulo 01 — LLMs: O Cérebro da Nova Geração de Aplicações

Imagine conversar com um computador como se fosse uma pessoa, fazer perguntas complexas e obter respostas detalhadas, escritas de forma natural. Isso parecia ficção científica até poucos anos atrás. Hoje, é possível graças aos **Large Language Models (LLMs)** — Modelos de Linguagem de Grande Porte. Eles são a base de ferramentas como ChatGPT, Claude, LLaMA, e de soluções que estamos prestes a construir com RAG.

Mas o que exatamente é um LLM? Como ele funciona? E por que ele revolucionou a forma como interagimos com máquinas?

Resumo: neste capítulo você entenderá o que são LLMs, como funcionam, suas limitações práticas e por que são essenciais para arquiteturas RAG.

Sumário
1. O que é um LLM?
2. Como funciona por dentro
3. Tokens
4. Janela de contexto
5. Parâmetros de geração
6. Limitações e riscos
7. Teste prático (Ollama)
8. Por que LLMs importam para RAG
9. Exercício sugerido
10. Conclusão e próximos passos

---

## 1. O que é um LLM?

Um **LLM** é um modelo matemático treinado para **prever a próxima palavra em uma sequência de texto**. Sim, apenas isso: prever a próxima palavra. Mas quando essa tarefa simples é escalada para bilhões de parâmetros e treinada com trilhões de palavras, o resultado é um modelo que parece “entender” linguagem, responder perguntas, escrever código e até criar poesia.

Para entender a essência disso, pense em como você completa frases no dia a dia:
> “Ontem eu fui ao…”  
Sua mente automaticamente preenche “mercado”, “cinema” ou outro lugar, baseado no contexto. O LLM faz o mesmo, mas em uma escala gigantesca.

---

## 2. Como funciona por dentro

O segredo dos LLMs está na arquitetura chamada **Transformer**. Introduzida em 2017, ela trouxe um conceito crucial: **atenção**. Em vez de processar palavras em sequência linear (como modelos antigos), o Transformer avalia **quais partes do texto são mais relevantes para prever a próxima palavra**.

Durante o treinamento:

1. O modelo recebe enormes quantidades de texto.
2. Ele ajusta bilhões de parâmetros para minimizar erros na predição da próxima palavra.
3. Com o tempo, aprende padrões complexos: gramática, estilo, relações semânticas e até lógica básica.

O resultado? Um modelo capaz de gerar textos coerentes, mesmo em contextos que nunca viu.

---

## 3. Tokens: a unidade de pensamento do modelo

LLMs não trabalham com palavras como nós. Eles quebram o texto em **tokens**, que podem ser palavras inteiras, pedaços de palavras ou até caracteres isolados. Por exemplo:

- "gato" pode ser um único token.
- "gatos" pode ser dividido em "gato" e "s".
- "paralelismo" pode ser quebrado em várias partes.

Isso tem implicações práticas:

- **Custo:** cada token processado consome tempo e recursos.
- **Limite:** modelos têm um número máximo de tokens que conseguem "lembrar" de uma vez, chamado **janela de contexto**.

---

## 3.1 Mini‑guia prático de tokens (contador, custo e orçamento)
- Por que contar tokens importa: custo (tokens processados = custo) e limite de contexto. Planeje prompt + contexto com um budget de tokens.
- Regra prática rápida: em português, estime ~3–4 caracteres por token (varia por idioma/encoding). Use uma biblioteca tokenizer para contagem precisa (ex.: `tiktoken` para modelos compatíveis).

Exemplo rápido (Python, usando tiktoken):

```python
# Exemplo conceitual — instale tiktoken quando aplicável
from tiktoken import encoding_for_model

def count_tokens(text, model_name="gpt-4o"):
    enc = encoding_for_model(model_name)
    return len(enc.encode(text))

text = "Explique paralelismo em programação em uma frase."
print(count_tokens(text))
```

Dicas práticas:
- Reserve pelo menos 10–20% do budget para tokens de saída (resposta do modelo).
- Para RAG, some: tokens_prompt + tokens_context + tokens_expected_output <= model_max_context.
- Ferramentas úteis: `tiktoken`, `transformers` tokenizers, utilitários de LangChain para estimar tamanho do prompt.

---

## 4. Janela de contexto: a memória imediata do modelo

A janela de contexto define quanto o modelo consegue considerar de uma vez. Se um modelo tem limite de 8.000 tokens, isso significa que ele só consegue levar em conta aproximadamente 6.000 palavras de entrada (prompt + histórico). Tudo além disso é "esquecido".

Por que isso importa? Porque, em aplicações reais, você precisará decidir **o que cabe na memória do modelo**. No RAG, isso significa selecionar apenas os trechos mais relevantes da sua base de conhecimento para incluir no prompt.

---

## 5. Ajustando o comportamento: temperatura e outros parâmetros

Embora um LLM pareça mágico, ele é controlável através de parâmetros:

- **Temperatura:** controla a criatividade.  
	- `0.0` = respostas determinísticas (sempre iguais para a mesma pergunta).  
	- `0.8` = respostas criativas e variadas.
- **Top-k e Top-p:** ajustam a diversidade do vocabulário escolhido.

Para aplicações como RAG, geralmente usamos **temperatura baixa**, garantindo consistência nas respostas.

---

## 6. As limitações que ninguém pode ignorar

Por mais impressionantes que sejam, LLMs têm limites claros:

1. **Alucinação:** quando não sabem algo, podem inventar de forma convincente.
2. **Desatualização:** conhecem apenas o que foi usado no treinamento.
3. **Sensibilidade a prompts:** pequenas mudanças na forma de perguntar podem alterar drasticamente as respostas.

Essas limitações são exatamente o motivo de estarmos construindo sistemas que **ampliam os LLMs com dados confiáveis**. E é aí que o **RAG** entra: em vez de confiar apenas na “memória” do modelo, vamos alimentá-lo com fatos relevantes e atualizados no momento da consulta.

---

## 6.1 Debug checklist (quando o sistema dá respostas ruins)
- Isolar: reproduza a pergunta com e sem contexto recuperado. Se sem contexto o modelo responde diferente, o problema pode estar no retriever ou nos chunks.
- Verificar k do retriever: k muito alto traz ruído; muito baixo perde evidência. Experimente k=1..10.
- Checar truncamento: conte tokens do prompt; se passar do limite, o prompt pode estar sendo cortado.
- Validar metadados: fontes e timestamps ajudam a saber se o chunk é da versão correta do documento.
- Teste de similaridade: verifique se embeddings para frases equivalentes são realmente próximos (sanity check).
- Tentar reranker simples (cross‑encoder) se muitos chunks parecidos aparecem no topo.

## 6.2 Prompt template prático para RAG + fallback anti‑alucinação
Template recomendado (use placeholders `context` e `question`):

```
Você é um assistente técnico. Use APENAS o contexto fornecido para responder.

CONTEXTO:
{context}

PERGUNTA: {question}

INSTRUÇÕES:
- Responda com precisão e objetividade.
- Cite as fontes/interno (ex.: fonte: documento.pdf, capítulo 3).
- Se a informação não estiver no contexto, responda exatamente: "Não encontrei essa informação no contexto fornecido."
- Não invente dados ou números.
```

Uso prático:
- Forme o `context` concatenando os N chunks mais relevantes (com metadados/fonte).
- Combine o template com temperatura baixa (ex.: 0.0–0.2) para reduzir variação.

---

## 7. Um teste prático com um LLM local

Para começar a brincar com um LLM local, vamos usar o [Ollama](https://ollama.com). Com ele rodando, digite:

```bash
ollama run llama3 "Explique paralelismo em programação em uma frase."
```

Você deve ver algo assim:

Paralelismo é a execução de múltiplas tarefas ao mesmo tempo para aumentar a eficiência.

Agora, rode com temperatura maior:

```bash
ollama run llama3 --temperature 0.8 "Explique paralelismo em programação em uma frase."
```

Compare as respostas. Na segunda, o modelo pode usar palavras diferentes ou até exemplos criativos. Isso ilustra como os parâmetros afetam o comportamento.

---

## 8. Por que entender LLMs é essencial para RAG?

Você poderia simplesmente jogar todos os seus documentos em um LLM e pedir para ele “ler tudo”. Mas isso não funciona:

- Custaria muito caro (tokens = dinheiro).
- Ultrapassaria limites de contexto.
- Geraria respostas possivelmente erradas (alucinações).

Ao compreender como um LLM funciona — especialmente suas forças (geração coerente) e fraquezas (memória limitada, desatualização) —, você estará pronto para aproveitar ao máximo o RAG, que combina recuperação inteligente com geração controlada.

---

## 8.1 Escalabilidade prática (quando passar do protótipo)
- Cache de respostas: armazene respostas a consultas frequentes para reduzir custo e latência.
- Batch de embeddings: processe documentos em lote para aproveitar GPUs/throughput.
- Reranking: inclua um reranker leve (cross‑encoder) para melhorar precisão antes de gerar o prompt final.
- Serviço de serving: para produção, considere migrar o endpoint de serving para uma stack mais performática (containers, Golang/Java para alta concorrência) mantendo Python para pipeline offline.

---

## 9. Exercício sugerido

1. Pergunte a um LLM sobre algo muito recente (últimos 3 meses).
2. Veja se ele responde com confiança, mesmo sem saber.
3. Pesquise a resposta verdadeira e compare.

Você perceberá como um modelo pode parecer certo… e ainda assim estar completamente errado. Esse é o problema que vamos resolver com RAG.

---

## 9.1 Métricas e avaliação rápida (como medir a qualidade)
- Precisão simples: proporção de respostas factuais corretas em um conjunto de perguntas de referência.
- Precisão@k (para retrieval): se a resposta correta está entre os top‑k chunks recuperados.
- Taxa de citação correta: frequência com que as fontes citadas na resposta realmente suportam a afirmação.
- Avaliação humana/QA: revisão manual de amostras para medir 'faithfulness'.

Como montar um teste simples:
1. Crie 20–50 perguntas com respostas de referência (gold).  
2. Rode o pipeline RAG e armazene: pergunta, resposta, fontes, chunks usados.  
3. Calcule métricas básicas (precision, precision@k) e registre num CSV para comparação de configurações.

---

## 9.2 Experimentos práticos sugeridos
- Teste A (temperatura): compare resultados com temperature=0.0 vs 0.8 em 5–10 perguntas factuais. Avalie variação e tendência a alucinações.
- Teste B (retrieval k): experimente k=1,3,5 e compare precision@k e qualidade das citações.
- Teste C (chunk size): compare chunk_size=500,1000,1500 tokens e observe impacto em recall/precisão.

Registro de resultados (exemplo CSV):
```
question,config,temp,k,chunk_size,answer,precision,precision@k,sources
"O que é paralelismo?","default",0.0,3,1000,"...",1,1,"manual.txt#p12"
```

---

## Apêndice: comandos úteis do Ollama e healthcheck
- Testar modelo local:

```bash
ollama run llama3 "Explique paralelismo em programação em uma frase."
```

- Testar com temperatura:

```bash
ollama run llama3 --temperature 0.8 "Explique paralelismo em programação em uma frase."
```

- Healthcheck / tags:

```bash
curl http://localhost:11434/api/tags
```

---

## Referências e leituras rápidas
- Paper Transformer (Vaswani et al., 2017)
- tiktoken / tokenizers (para contagem de tokens)
- LangChain prompts guide
- Artigos sobre RAG e Grounding (AWS Bedrock, Vertex RAG Engine docs)