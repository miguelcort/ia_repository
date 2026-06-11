# Walkthrough de DeepSeek-V3

> DeepSeek-V3 (Feb 2025): 671B MoE total, 37B activos (256 routed + 2 shared experts, top-8 routing). Innovations: MLA (Multi-Latent Attention, compressed KV, -90% memory), NSA (Native Sparse Attention, compression + selection + sliding, 12x speedup), DualPipe (bidirectional pipeline, 2x overlap compute/comm), FP8 training, MTP (Multi-Token Prediction, +5x training signal), 14.8T tokens training, 2.788M H800 GPU-hours, ~$5-10M cost (1/10 de closed frontier). Comparable o > Llama 3.1 405B en benchmarks. Variants: V3 Base, V3 Instruct, V3 R1 (reasoning). MIT license, open source. Frameworks: custom kernels, vLLM support.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 10/05-scaling-y-distribuido, 10/11-cuantizacion, 10/17-atencion-nativa-dispersa, 10/19-dualpipe-y-paralelismo
**Tiempo estimado:** ~30 minutos

## Objetivos

- Listar components de DeepSeek-V3.
- Diagnosticar MoE routing (256 experts, top-8, 2 shared).
- Calcular active params (37B/671B).
- Comparar costos vs closed frontier.

## Constrúyelo

```python
def moe_routing(top_k=8, n_experts=256, n_shared=2):
    return top_k + n_shared
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-deepseek-v3
fase: 10
leccion: 20
---

1. 671B MoE, 37B active.
2. MLA, NSA, DualPipe.
3. FP8, MTP, $5-10M.
4. 14.8T tokens.
5. MIT, open source.
```

## Ejercicios

1. **DeepSeek-V3**: deploy
   DeepSeek-V3 con vLLM.
2. **MLA**: implementar MLA
   simplificado.
3. **Desafio**: fine-tune
   DeepSeek-V3 con LoRA.

## Lecturas recomendadas

- "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" (DeepSeek-AI, 2024)
- "DeepSeek-V3 Technical Report" (DeepSeek-AI, 2025)
- "Multi-Latent Attention (MLA)" (DeepSeek, 2024)
- "Native Sparse Attention (NSA)" (DeepSeek-AI, 2025)

---

> 📚 **Adaptación al español** de la lección "[DeepSeek V3 Walkthrough]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).