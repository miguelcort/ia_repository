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

## Ejercicios

1. **MTP**: implementar MTP training en Llama 3
   (modificar Hugging Face model).
2. **Spec decode**: usar aux heads para speculative
   decoding con EAGLE3.
3. **Desafío**: MTP + EAGLE3 combo, medir speedup
   en HumanEval+.

## MTP architectures y trade-offs

MTP (Gloeckle 2024, Meta): cada capa del transformer
añade K+1 prediction heads desde un trunk compartido.
Main head predice t+1 (siguiente token), aux heads
predicen t+2, ..., t+K+1. Los K aux heads comparten
parámetros del trunk pero tienen capas lineales
independientes. Training: CE(main) + aux_weight *
Σ CE(aux), aux_weight=0.3 típico.

Ventajas cuantificadas: (1) +5x training signal
density (5 heads vs 1). (2) +5-10% en benchmarks
downstream. (3) Spec decoding 2-3x speedup: aux heads
son drafts, main head verifica, accept tokens que
coinciden. (4) Mejor data efficiency: el modelo
aprende dependencias de largo alcance.

Implementaciones: (1) Meta MTP (research repo). (2)
DeepSeek-V3 (2025, +5% benchmarks, 2x throughput
inference). (3) EAGLE3 (Li 2024) usa MTP-style
features para spec decoding, 2-3x speedup. (4)
Medusa (Cai 2024) similar pero heads independientes.

Trade-offs: (1) +5% compute en training (K heads). (2)
Memory overhead: K+1 logits por posición. (3) No
mejora para generation tasks donde solo importa
logit[t+1]. (4) Aux heads pueden no converger si
aux_weight es muy bajo.

Spec decoding: para cada step, generar K candidatos
con aux heads en paralelo, verificar con main head
(1 forward pass), aceptar longest prefix matching.
Speedup = 2-3x en Llama 3 70B + EAGLE3. Compatible
con KV cache, paged attention. Frameworks: EAGLE
(THU), Medusa (MIT), SpecInfer (Microsoft).

MTP es SOTA para spec decode en frontier models 2024+
(Llama 3+, DeepSeek-V3, Qwen 2.5). Usar aux_weight=0.3,
K=4 heads, learning rate 0.95x el del main head.

## Lecturas recomendadas

- "Better & Faster Large Language Models via Multi-token Prediction" (Gloeckle et al., 2024)
- "DeepSeek-V3 Technical Report" (DeepSeek-AI, 2025)
- "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test" (Li et al., 2024)
- "Multi-Token Prediction" (Meta, 2024)
- "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" (Cai et al., 2024)

---

> 📚 **Adaptación al español** de la lección "[Multi Token Prediction]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).