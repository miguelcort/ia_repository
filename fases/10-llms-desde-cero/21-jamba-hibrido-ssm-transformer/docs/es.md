# Jamba híbrido SSM transformer

> Jamba (AI21 2024): hybrid Mamba + transformer, 8 transformer + 28 mamba blocks alternating. Jamba 1.0 (12B) y Jamba 1.5 Large (52B MoE). Context: 256K tokens. Mamba: state-space model (SSM), O(n) compute, O(d · state) memory (vs attention O(n²)). Best of both: Mamba efficiency + Transformer quality. Mamba-2: state-space duality (SSD). Variantes: Jamba, Zamba (Zyphra), RecurrentGemma (Google Griffin). Hoy: hybrid es SOTA para long context. Frameworks: mamba-ssm, mamba.py, Hugging Face. Open weights Apache 2.0.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07/15-atencion-variantes, 10/17-atencion-nativa-dispersa
**Tiempo estimado:** ~30 minutos

## Objetivos

- Listar components de Jamba architecture.
- Comparar Mamba vs Transformer complexity.
- Diagnosticar hybrid ratios.
- Diagnosticar trade-offs.

## Constrúyelo

```python
def ssm_complexity(seq_len, dim, state_size=16):
    compute = seq_len * dim * state_size
    memory = dim * state_size
    return compute, memory
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-jamba
fase: 10
leccion: 21
---

1. Jamba: 8 transformer + 28 mamba.
2. 12B / 52B MoE, 256K context.
3. Mamba O(n) vs attention O(n^2).
4. Hybrid best of both.
5. Mamba-2 SSD, Jamba 1.5 reasoning.
```

## Ejercicios

1. **Mamba**: implementar SSM
   simple.
2. **Jamba**: deploy Jamba 12B
   con vLLM.
3. **Desafio**: hybrid Mamba +
   transformer custom.

## Lecturas recomendadas

- "Jamba: A Hybrid Transformer-Mamba Language Model" (AI21, 2024)
- "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)
- "Mamba-2: SSD" (Dao, 2024)
- "RecurrentGemma: Griffin" (Google DeepMind, 2024)

---

> 📚 **Adaptación al español** de la lección "[Jamba Hybrid SSM Transformer]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).