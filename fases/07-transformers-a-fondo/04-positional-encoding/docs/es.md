# Positional encoding

> Self-attention es permutation-invariant. PE inyecta orden: sinusoidal (Transformer original), aprendido (BERT), RoPE (Llama, Mistral, Qwen, Phi-3), ALiBi (BLOOM, MPT). RoPE rota pares de dimensiones, ALiBi sesgo lineal por distancia. Extensión de contexto: PI, NTK-aware, YaRN.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07/03-multi-head-attention
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar sinusoidal PE.
- Implementar RoPE con rotación por pares.
- Implementar ALiBi con bias lineal.
- Comparar propiedades de extrapolación.

## Constrúyelo

```python
def sinusoidal_pe(seq_len, d_model):
    pos = np.arange(seq_len).reshape(-1, 1)
    i = np.arange(d_model).reshape(1, -1)
    angle = pos / (10000 ** (2 * (i // 2) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-positional-encoding
fase: 07
leccion: 04
---

1. Sin PE, attention es set-based.
2. RoPE: <q_rot(m), k_rot(n)> depende solo de m-n.
3. ALiBi: bias = -slope_h * distancia.
4. Sinusoidal: freq geometricas, no aprendido.
5. Extension: PI, NTK-aware, YaRN.
```

## Ejercicios

1. **Visualizacion**: graficar PE(pos, dim) para
   dim=0, 1, d/2, d-1.
2. **RoPE vs sinusoidal**: entrenar transformer
   pequeno y comparar PPL.
3. **Desafio**: implementar YaRN (NTK + attention
   scaling) y evaluar extension de contexto.

## Lecturas recomendadas

- "Attention Is All You Need" (Vaswani et al., 2017)
- "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
- "Train Short, Test Long: Attention with Linear Biases" (Press et al., 2022)
- "YaRN: Efficient Context Window Extension of Large Language Models" (Peng et al., 2023)

---

> 📚 **Adaptación al español** de la lección "[Positional Encoding]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).