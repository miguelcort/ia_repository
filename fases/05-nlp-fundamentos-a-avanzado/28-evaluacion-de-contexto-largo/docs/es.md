# 28 — Evaluación de contexto largo: NIAH, RULER, LongBench, MRCR

> Los LLMs modernos soportan contextos de 128k+ tokens. Pero ¿realmente usan esa capacidad? Evaluaciones como Needle-in-a-Haystack, RULER, y LongBench miden la capacidad real de contexto largo.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 27-frameworks-de-evaluacion-de-llm
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar NIAH (Needle-in-a-Haystack) para medir
  recuperación de información a varias profundidades.
- Aplicar RULER, LongBench, y MRCR para evaluación
  comprehensiva.
- Diagnosticar las limitaciones de contexto largo de
  diferentes LLMs.
- Conectar a RAG y agentic long-context tasks.

## El problema

GPT-4 Turbo tiene 128k tokens, Claude 3.5 200k, Gemini
1.5 1M. Pero los modelos no usan el contexto completo
eficientemente: pierden información en el medio
("lost in the middle"), olvidan el principio, o
confunden información de diferentes partes. La lección
cubre los benchmarks que miden realmente la capacidad de
contexto largo.

## El concepto

**Needle-in-a-Haystack (NIAH).** Benchmark simple de
Greg Kamradt. Inserta un hecho aleatorio ("la mejor
comida es el mole") en una posición aleatoria de un
contexto largo (Paul Graham essays). Mide si el LLM lo
recupera. Se varía:
- Posición (inicio, medio, final).
- Longitud del contexto (1k, 10k, 100k, 1M).

Revela "lost in the middle": los LLMs son mejores al
principio y al final que en el medio.

**RULER (Hsieh et al., 2024).** Extensión de NIAH con
más tareas: recuperación de múltiples needles, tracing
de variables (similar a la aguja en un pajar pero con
información estructurada), multi-hop reasoning, etc.
Más discriminativo que NIAH.

**LongBench (Bai et al., 2023).** 21 tareas en chino e
inglés que requieren contexto largo: QA sobre documentos,
summarization, code completion, few-shot learning.
Promediado en 0-100 score.

**MRCR (Multi-Round Co-reference Resolution).** Mide
si el modelo puede mantener el estado de una
conversación larga y resolver correferencias a lo largo
de múltiples turnos. Especialmente relevante para
agentes.

**Benchmarks recientes.**

- **LongBench v2 (2024):** versión más desafiante de
  LongBench.
- **RULER++ (2025):** tareas más difíciles y largas.
- **NoLiMa (2024):** mide "no lost in middle".
- **LEval (2023):** long-context evaluation.

**Cómo interpretar los resultados.**

- NIAH accuracy 100% en el principio y al final, ~70%
  en el medio = "lost in the middle" clásico.
- RULER score < 50% en longitudes grandes = el modelo
  no usa el contexto eficientemente.
- LongBench score < 30 = el modelo no es apto para
  tareas de contexto largo.

**Aplicaciones.**

- **RAG con muchos chunks:** ¿realmente usa el LLM los
  chunks del medio? Medir con NIAH.
- **Agentes con historiales largos:** ¿el agente
  recuerda instrucciones de turnos anteriores?
- **Análisis de documentos:** ¿el LLM puede resumir un
  paper de 100 páginas fielmente?

**Trampas.**

- **Benchmarks sintéticos vs reales:** NIAH inserta
  facts random; el mundo real tiene dependencies
  semánticas. Usar benchmarks con texto natural
  (LongBench) además de NIAH.
- **Posición del needle:** muchos papers solo prueban
  inserción al final. Probar todas las posiciones.
- **Longitud del contexto:** los modelos degradan
  antes de su límite oficial. Medir a varias
  longitudes.

## Constrúyelo

```python
import random
import re


def needle_in_haystack(context, needle, question):
    """Mide si el LLM recupera el needle del contexto.
    context: texto largo, needle: hecho a insertar,
    question: pregunta sobre el needle."""
    # Insertar needle en posición aleatoria
    pos = random.randint(0, len(context))
    augmented = context[:pos] + f"\n\n{needle}\n\n" + context[pos:]
    prompt = (
        f"Contexto:\n{augmented}\n\n"
        f"Pregunta: {question}\n"
        f"Respuesta:"
    )
    return prompt


def evaluate_niah(llm_fn, n_trials=10, lengths=(1000, 5000, 10000)):
    """Evalúa NIAH a varias longitudes."""
    needles = [
        "La mejor comida del mundo es el mole poblano.",
        "El número mágico es 4287.",
        "El libro favorito del autor es 'Cien años de soledad'.",
    ]
    results = {}
    for length in lengths:
        # Generar contexto dummy
        context = " ".join(
            ["Esta es una oración de relleno."] * (length // 5)
        )
        scores = []
        for trial in range(n_trials):
            needle = random.choice(needles)
            question = "¿Cuál es el dato relevante?"
            prompt = needle_in_haystack(context, needle, question)
            response = llm_fn(prompt)
            # ¿Recupera el needle?
            recovered = any(
                word in response.lower()
                for word in needle.lower().split()[:5]
            )
            scores.append(1.0 if recovered else 0.0)
        results[length] = sum(scores) / len(scores)
    return results
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-long-context-eval
fase: 05
leccion: 28
---

Eres un asistente que ayuda a evaluar la capacidad de
contexto largo. Recibirás el modelo y la aplicación. Tu
trabajo:

1. Para diagnosticar "lost in the middle": NIAH con
   inserción en inicio, 25%, 50%, 75%, final.
2. Para evaluación comprehensiva: RULER.
3. Para tareas de contexto largo realistas: LongBench
   v2.
4. Para agentes multi-turno: MRCR.
5. Probar a varias longitudes: 1k, 10k, 50k, 100k,
   200k.
6. Comparar modelos en el mismo set de pruebas.
7. Recomienda RAG cuando el contexto es muy largo y
   el LLM no retiene.
8. Advertir contra confiar en la longitud máxima
   reportada: el modelo degrada antes.
```

## Ejercicios

1. **NIAH**: implementa y evalúa un LLM en contextos
   de 1k, 10k, 100k.
2. **RULER**: aplica RULER a un LLM y compara con
   otro.
3. **Desafío**: diseña un benchmark custom para tu
   tarea de RAG.

## Lecturas recomendadas

- *Needle in a Haystack* — Greg Kamradt, 2023.
- *RULER* — Hsieh et al., 2024.
- *LongBench* — Bai et al., 2023.
- *Lost in the Middle* — Liu et al., 2023.

---

> 📚 **Adaptación al español** de la lección "[Long Context Evaluation]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
