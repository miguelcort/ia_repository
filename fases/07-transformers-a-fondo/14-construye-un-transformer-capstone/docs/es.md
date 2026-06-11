# Construye un transformer (capstone)

> Capstone: decoder-only transformer completo en numpy. Token embedding + sinusoidal PE + N decoder blocks (causal self-attn + FFN, Pre-LN) + LM head con weight tying. Forward, backward, AdamW training, lr cosine schedule, mixed precision, top-k/top-p sampling. ~150 líneas de código. Integra todo: attention, FFN, PE, masking, generation, weight tying.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07/05-transformer-completo
**Tiempo estimado:** ~45 minutos

## Objetivos

- Construir decoder-only transformer.
- Implementar weight tying.
- Forward pass completo con masking.
- Training loop con AdamW.

## Constrúyelo

```python
class TransformerLM:
    def forward(self, token_ids):
        x = self.token_emb[token_ids]
        x = x + self.pos_emb[:len(token_ids)]
        mask = causal_mask(len(token_ids))
        for block in self.blocks:
            x = block(x, mask)
        return x @ self.W_head.T
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-transformer-capstone
fase: 07
leccion: 14
---

1. Decoder: emb + PE + N blocks + head.
2. Weight tying: W_head = W_emb.T.
3. AdamW, cosine, wd 0.1.
4. T + top_k + top_p sampling.
5. bf16 mixed precision.
```

## Ejercicios

1. **Tiny GPT**: entrenar transformer de
   6 capas en un corpus de texto.
2. **Beam search**: implementar beam decoding
   con length normalization.
3. **Desafio**: implementar RoPE y reentrenar,
   comparar quality.

## Lecturas recomendadas

- "Attention Is All You Need" (Vaswani et al., 2017)
- "Language Models are Few-Shot Learners" (Brown et al., 2020)
- "nanoGPT" (Karpathy, 2022) - minimal GPT training
- "minbpe" (Karpathy) - minimal BPE tokenizer

---

> 📚 **Adaptación al español** de la lección "[Build a Transformer Capstone]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).