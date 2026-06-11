# DualPipe y paralelismo

> Schedules de pipeline: GPipe (all-forward-then-backward, +bubble), 1F1B (one-forward-one-backward, standard, bubble=(S-1)/M), Interleaved (PipeDream, chunks V, -bubble), ZB-H1/H2 (zero bubble), DualPipe (DeepSeek 2025, bidirectional, forward+backward simultaneous, 2x overlap compute/comm, bubble~0). Bubble comparison: GPipe > 1F1B > Interleaved > ZB-H1 > DualPipe. DeepSeek-V3 671B usa DualPipe + NSA + MLA + MoE, 14.8T tokens, 2.788M H800 GPU-hours. Frameworks: Megatron-Core (1F1B, Interleaved, ZB-H1), DeepSpeed, custom. Hoy: 1F1B production, DualPipe frontier.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 10/05-scaling-y-distribuido
**Tiempo estimado:** ~30 minutos

## Objetivos

- Calcular bubble fractions.
- Comparar 1F1B, Interleaved, ZB-H1, DualPipe.
- Diagnosticar trade-offs.
- Identificar SOTA.

## Constrúyelo

```python
def pipeline_bubble_fraction(n_microbatches, n_stages):
    return (n_stages - 1) / n_microbatches
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-dualpipe
fase: 10
leccion: 19
---

1. GPipe, 1F1B, Interleaved.
2. ZB-H1, DualPipe.
3. Bubble comparison.
4. 2x overlap compute/comm.
5. DeepSeek-V3 frontier.
```

## Ejercicios

1. **Pipeline**: implementar 1F1B
   en toy model.
2. **Bubble**: comparar 1F1B
   vs Interleaved vs DualPipe.
3. **Desafio**: configurar
   DualPipe en cluster 16 GPUs.

## Lecturas recomendadas

- "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism" (Huang et al., 2019)
- "PipeDream: Generalized Pipeline Parallelism for DNN Training" (Narayanan et al., 2021)
- "DeepSeek-V3 Technical Report" (DeepSeek-AI, 2025)
- "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (Shoeybi et al., 2019)

---

> 📚 **Adaptación al español** de la lección "[DualPipe Parallelism]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).