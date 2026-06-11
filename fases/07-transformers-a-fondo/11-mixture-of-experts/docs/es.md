# Mixture of experts

> MoE: N experts (FFNs) por capa, router aprende a asignar tokens a top-k experts. Sparse activation (k/N expertos activos) con dense params (todos almacenados). Load balancing loss evita colapso a pocos experts. Variantes: Switch (top-1), Mixtral 8x7B (top-2), GLaM (top-2, 64 experts), DeepSeek-V2 (fine-grained, 160 experts).

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07/05-transformer-completo
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar top-k routing.
- Implementar MoE layer con expert dispatch.
- Calcular load balancing loss.
- Diagnosticar trade-offs MoE vs denso.

## Constrúyelo

```python
def top_k_routing(x, W_gate, n_experts, top_k):
    logits = x @ W_gate
    probs = softmax(logits, axis=-1)
    topk_idx = np.argpartition(-probs, top_k, axis=-1)[..., :top_k]
    topk_w = np.take_along_axis(probs, topk_idx, axis=-1)
    topk_w = topk_w / topk_w.sum(axis=-1, keepdims=True)
    return topk_idx, topk_w
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-moe
fase: 07
leccion: 11
---

1. N experts, router, top-k.
2. Aux loss: balance.
3. Switch (top-1), Mixtral (top-2).
4. Capacity factor, expert dropeo.
5. Memory: dense, compute: sparse.
```

## Ejercicios

1. **Comparar**: Mixtral-style top-2 vs top-1
   en quality y compute.
2. **Fine-grained**: implementar MoE con 64+
   experts.
3. **Desafio**: implementar capacity factor
   con dropeo de tokens.

## Lecturas recomendadas

- "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (Fedus et al., 2022)
- "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding" (Lepikhin et al., 2020)
- "Mixtral of Experts" (Jiang et al., 2024)

---

> 📚 **Adaptación al español** de la lección "[Mixture of Experts]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).