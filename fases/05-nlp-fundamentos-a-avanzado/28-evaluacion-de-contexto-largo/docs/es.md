# Evaluación de contexto largo

> Medir la capacidad del LLM de usar contexto largo. Benchmarks: Needle-in-a-Haystack, RULER, LongBench. Gemini 1.5 (1M), Claude 3.5 (200K), GPT-4 (128K). RAG + long context es el approach hibrido. Chunking sigue siendo el default.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 27-frameworks-de-evaluacion-de-llm
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar Needle-in-a-Haystack.
- Calcular posicion del needle.
- Diagnosticar NIAH vs RULER.
- Diagnosticar context window optimo.

## Constrúyelo

```python
def needle_haystack_test(respuesta, needle, contexto):
    needle_lower = needle.lower()
    resp_lower = respuesta.lower()
    return 1.0 if needle_lower in resp_lower else 0.0
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-long-context
fase: 05
leccion: 28
---

1. Default: 8-32K, chunking + RAG.
2. Long doc QA: Claude 3.5 200K, GPT-4 128K.
3. Novel-length: Gemini 1.5 1M.
4. Eval: NIAH + RULER + LongBench.
5. Hybrid: RAG + long context para critico.
```

## Ejercicios

1. **NIAH completo**: implementar NIAH con diferentes
   profundidades y context lengths.
2. **RULER subset**: implementar CWE, FWE, multi-hop de
   RULER.
3. **Desafio**: comparar GPT-4, Claude 3.5, Gemini 1.5
   en RULER, identificar donde cada uno falla.

## Lecturas recomendadas

- "RULER" (Hsieh et al., 2024)
- "LongBench" (Bai et al., 2023)
- "YaRN" (Peng et al., 2023)

---

> 📚 **Adaptación al español** de la lección "[Long Context Evaluation]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).