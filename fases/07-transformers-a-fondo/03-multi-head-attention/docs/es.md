# Multi-head attention

> Multi-head attention: h cabezas en paralelo, cada una con sus proyecciones Q, K, V. d_k = d_model / h. Concatenar + W_O. Cada cabeza aprende patrón distinto (sintáctico, semántico, posicional, coreferencia). Reducciones de KV cache: MQA (1 KV) o GQA (g grupos).

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07/02-self-attention-desde-cero
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar split/merge de heads.
- Implementar multi-head attention con h cabezas.
- Diagnosticar trade-offs MHA/MQA/GQA.
- Visualizar patrones aprendidos.

## Constrúyelo

```python
def multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads, mask=None):
    d_model = X.shape[-1]
    d_k = d_model // n_heads
    Q = (X @ W_Q).reshape(-1, n_heads, d_k).transpose(1, 0, 2)
    K = (X @ W_K).reshape(-1, n_heads, d_k).transpose(1, 0, 2)
    V = (X @ W_V).reshape(-1, n_heads, d_k).transpose(1, 0, 2)
    head_outs = [self_attention(Q[h], K[h], V[h], mask)[0]
                 for h in range(n_heads)]
    concat = np.stack(head_outs).transpose(1, 0, 2).reshape(-1, d_model)
    return concat @ W_O
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-multi-head-attention
fase: 07
leccion: 03
---

1. h cabezas paralelas, d_k = d / h.
2. MHA, MQA, GQA segun memoria disponible.
3. BertViz, exBERT para visualizar.
4. Concat + W_O.
5. Trade-off: mas heads = mas fino, mas
   redundancy.
```

## Ejercicios

1. **MQA**: convertir MHA a MQA, comparar PPL y
   memoria de KV cache.
2. **Visualizacion**: usar BertViz para ver que
   atiende cada head en una capa.
3. **Desafio**: implementar GQA con n grupos y comparar
   vs MHA.

## Lecturas recomendadas

- "Attention Is All You Need" (Vaswani et al., 2017)
- "Multi-Query Attention" (Shazeer, 2019)
- "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (Ainslie et al., 2023)

---

> 📚 **Adaptación al español** de la lección "[Multi-Head Attention]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).