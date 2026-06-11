# Traducción automática

> Traducir texto entre idiomas. SMT (2010s) -> NMT con attention (2015) -> Transformer (2017) -> LLMs zero-shot. SOTA open: NLLB-200 (200 idiomas). SOTA closed: GPT-4, Claude, Gemini, Google Translate. BLEU + COMET + human eval son las metricas.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 10-mecanismo-de-atencion
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar BLEU score con n-gramas y brevity penalty.
- Diagnosticar SMT vs NMT vs LLM zero-shot.
- Evaluar MT en produccion.

## Constrúyelo

```python
def bleu_score(reference, candidate, max_n=4):
    # n-gram precision + brevity penalty
    ...
    return bp * np.exp(log_bleu)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-mt
fase: 05
leccion: 11
---

1. Pares comunes: NLLB-200, zero-shot GPT-4/Claude.
2. Volumen: NLLB distilled, Google Translate API.
3. Pares raros: NLLB fine-tune 5-50K pares.
4. Custom: NLLB + glossary.
5. Open sin infra: OPUS-MT.
6. BLEU + chrF + COMET + human eval.
```

## Ejercicios

1. **chrF**: implementar character n-gram F-score.
2. **COMET**: usar modelo unbabel-comet para evaluacion
   automatica.
3. **Desafio**: fine-tunear NLLB-200 distilled en 10K
   pares paralelos en-es de un dominio custom, mejorar
   chrF en +5.

## Lecturas recomendadas

- "BLEU" (Papineni et al., 2002)
- "NLLB" (Meta, 2022): <https://github.com/facebookresearch/flores>
- sacrebleu: <https://github.com/mjpost/sacrebleu>

---

> 📚 **Adaptación al español** de la lección "[Machine Translation]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).