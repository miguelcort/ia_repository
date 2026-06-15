# 27 — Frameworks de evaluación de LLMs

> Evaluar LLMs requiere más que accuracy. Necesitas métricas para razonamiento, factuality, safety, bias, y dominios específicos. Frameworks como RAGAS, DeepEval, y G-Eval simplifican esto.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11-ingenieria-llms
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Implementar métricas de evaluación: faithfulness,
  answer relevance, context precision/recall.
- Aplicar frameworks: RAGAS, DeepEval, G-Eval, MLflow
  LLM Evaluate.
- Diagnosticar cuándo usar cada métrica.
- Diseñar un benchmark de evaluación para tu tarea.

## El problema

Evaluar LLMs es difícil. Métricas clásicas como accuracy
capturan poco: el LLM puede tener la respuesta correcta
en el sentido equivocado, o alucinar. Necesitas
métricas específicas por capacidad: faithfulness (¿la
respuesta está en el contexto?), answer relevance
(¿responde la pregunta?), harmfulness (¿es segura?),
bias (¿trata a todos los grupos por igual?). Los
frameworks modernos automatizan esto.

## El concepto

**Tipos de métricas.**

- **Métricas clásicas:** accuracy, F1, BLEU, ROUGE.
  Útiles para tareas con respuestas específicas.
- **Métricas de LLM-as-judge:** un LLM (típicamente
  GPT-4) evalúa las respuestas de otro LLM. Powerful
  pero con sesgos.
- **Métricas de NLP:** BERTScore, BARTScore. Más robustas
  que BLEU/ROUGE.
- **Métricas de dominio:** precisión médica, exactitud
  legal, etc.

**Métricas RAG (RAGAS).**

- **Faithfulness:** ¿la respuesta está soportada por
  el contexto? (LLM-as-judge: el LLM verifica cada
  claim).
- **Answer relevance:** ¿la respuesta es relevante a la
  pregunta? (LLM-as-judge).
- **Context precision:** ¿los chunks recuperados son
  relevantes?
- **Context recall:** ¿el contexto cubre la respuesta
  ground truth?

**Frameworks populares.**

- **RAGAS (Es et al., 2023):** métricas para RAG.
  Faithfulness, answer relevance, context precision/
  recall.
- **DeepEval (Confident AI):** 30+ métricas. Incluye
  G-Eval (LLM-as-judge con criterios custom).
- **G-Eval (Liu et al., 2023):** LLM-as-judge con
  chain-of-thought. SOTA correlation con humanos.
- **MLflow LLM Evaluate:** parte de MLflow. Métricas
  predefinidas + custom.
- **Promptfoo:** testing de prompts con métricas
  automatizadas.
- **LangSmith:** parte de LangChain. Tracing y
  evaluación.

**Diseño de un benchmark de evaluación.**

1. **Construir un dataset de evaluación** (golden set):
   - 100-1000 ejemplos anotados manualmente.
   - Cubrir casos fáciles, difíciles, y edge cases.
2. **Definir métricas:** qué medir (faithfulness,
   answer relevance, latency, cost).
3. **Comparar versiones:** cambios en prompt, modelo,
   parámetros.
4. **Monitorear en producción:** muestras de uso real
   para detectar drift.

**Trampas.**

- **Confiar en una sola métrica:** siempre usa
  múltiples (faithfulness + relevance + latency).
- **LLM-as-judge sesgado:** GPT-4 prefiere respuestas
  más largas, en su estilo. Cross-validar con humanos
  en muestras pequeñas.
- **Benchmark demasiado fácil:** si el modelo logra
  100% en tu benchmark, es demasiado fácil.
- **Sin ground truth:** sin anotaciones humanas, no
  puedes calcular métricas supervisadas.

## Constrúyelo

```python
from collections import Counter
import re


def faithfulness(answer, context):
    """¿Cada claim en la respuesta está en el contexto?
    Simplificado: busca tokens clave."""
    answer_tokens = set(re.findall(r"\w+", answer.lower()))
    context_tokens = set(re.findall(r"\w+", context.lower()))
    # Quitar stopwords básicos
    stopwords = {"el", "la", "de", "en", "y", "a", "que", "es", "un", "una"}
    answer_tokens -= stopwords
    return len(answer_tokens & context_tokens) / max(len(answer_tokens), 1)


def answer_relevance(question, answer):
    """¿La respuesta responde a la pregunta?
    Simplificado: overlap de tokens clave."""
    q_tokens = set(re.findall(r"\w+", question.lower()))
    a_tokens = set(re.findall(r"\w+", answer.lower()))
    stopwords = {"el", "la", "de", "en", "y", "a", "que", "es", "un", "una"}
    q_tokens -= stopwords
    a_tokens -= stopwords
    return len(q_tokens & a_tokens) / max(len(q_tokens), 1)


def llm_as_judge(prompt, response, criterion, judge_model_fn):
    """LLM-as-judge: el LLM evalúa la respuesta según un
    criterio."""
    eval_prompt = (
        f"Criterio: {criterion}\n"
        f"Prompt: {prompt}\n"
        f"Respuesta: {response}\n"
        f"¿Cumple el criterio? Responde con un JSON: "
        '{"score": <0-10>, "razon": "..."}'
    )
    result = judge_model_fn(eval_prompt)
    return result
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-llm-eval
fase: 05
leccion: 27
---

Eres un asistente que ayuda a evaluar LLMs. Recibirás la
tarea y el caso de uso. Tu trabajo:

1. Para RAG: RAGAS (faithfulness, answer relevance,
   context precision/recall).
2. Para chatbots: G-Eval con criterios custom.
3. Para clasificación de texto: accuracy, F1.
4. Para QA: EM, F1.
5. Para generación abierta: BERTScore o LLM-as-judge.
6. Construir un golden set de 100-1000 ejemplos.
7. Comparar versiones en el golden set.
8. Monitorear en producción con samples.
9. Cross-validar LLM-as-judge con humanos en
   muestras pequeñas.
```

## Ejercicios

1. **Faithfulness**: implementa el chequeo de cada
   claim contra el contexto.
2. **Golden set**: construye un dataset de evaluación
   para tu tarea.
3. **Desafío**: integra RAGAS en un pipeline de
   producción y mide latencia vs accuracy.

## Lecturas recomendadas

- *RAGAS* — Es et al., 2023.
- *G-Eval* — Liu et al., 2023.
- *DeepEval* — <https://github.com/confident-ai/deepeval>.
- *MLflow LLM Evaluate* — <https://mlflow.org>.
- *Promptfoo* — <https://promptfoo.dev>.

---

> 📚 **Adaptación al español** de la lección "[LLM Evaluation Frameworks]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
