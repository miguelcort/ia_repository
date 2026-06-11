# Multi-Token Prediction (MTP)

> MTP (Gloeckle 2024, Meta): predecir múltiples tokens futuros simultáneamente. Main head (t+1) + aux heads (t+2, ..., t+k+1) desde shared trunk. Loss: CE(main) + aux_weight · Σ CE(aux), aux_weight=0.3 típico. Beneficios: +5x training signal density, +5-10% en benchmarks, spec decoding 2-3x speedup via aux heads. Llama 3+, DeepSeek-V3 usan MTP. Compatible con EAGLE3, Medusa, draft model. Frameworks: Meta MTP, DeepSeek-V3. Hoy: MTP es SOTA para spec decode en frontier models.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 10/04-pre-training-mini-gpt
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar MTP loss.
- Calcular training signal gain.
- Calcular inference speedup.
- Diagnosticar MTP components.

## Constrúyelo

```python
def mtp_loss(main_logits, aux_logits, targets, aux_targets, aux_weight=0.3):
    main_loss = cross_entropy(main_logits, targets)
    aux_loss = cross_entropy(aux_logits, aux_targets)
    return main_loss + aux_weight * aux_loss
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-mtp
fase: 10
leccion: 18
---

1. Main + aux heads.
2. aux_weight 0.3.
3. +5x signal density.
4. Spec decode 2-3x.
5. Llama 3+, DeepSeek-V3.
```

## Ejercicios

1. **MTP**: implementar MTP
   training en Llama 3.
2. **Spec decode**: usar aux
   heads para spec decode.
3. **Desafio**: MTP + EAGLE3
   combo.

## Lecturas recomendadas

- "Better & Faster Large Language Models via Multi-token Prediction" (Gloeckle et al., 2024)
- "DeepSeek-V3 Technical Report" (DeepSeek-AI, 2025)
- "Multi-Token Prediction" (Meta, 2024)

---

> 📚 **Adaptación al español** de la lección "[Multi Token Prediction]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).