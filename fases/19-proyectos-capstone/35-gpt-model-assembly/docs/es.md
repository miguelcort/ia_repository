# 35 — GPT model assembly

> GPT model: tokenizer + embeddings + N transformer blocks + LM head (tied weights). Componentes: kv-cache, generation (greedy, top-k, top-p, temperature). nanoGPT (Karpathy 2022) es la referencia minimal.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/30-34
**Tiempo estimado:** ~30 minutos

## Objetivos

- Assembly completo del GPT.
- LM head tied.
- Generation strategies.
- nanoGPT style.

## Constrúyelo

```python
import torch
import torch.nn as nn


class GPT(nn.Module):
    def __init__(self, vocab, d, n_heads, n_layers, max_len=2048):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d, n_heads, 4 * d)
             for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-gpt
fase: 19
leccion: 35
---

1. Tokenizer + emb.
2. N transformer blocks.
3. LM head tied.
4. Generation.
5. nanoGPT ref.
```

## Ejercicios

1. **GPT-small**: 12 layers,
   12 heads.
2. **Generation**: greedy,
   top-k, top-p.
3. **Desafío**: 100M
   params model.

## Detalles de assembly

Tied weights: LM head weight = token embedding
weight. Reduce params ~d_model × vocab. Standard en
GPT-2/3, Llama, nanoGPT.

Generation strategies: (1) Greedy: argmax. (2)
Top-k: filter top K probs, renormalize, sample. K=50
típico. (3) Top-p (nucleus): smallest set with
cumulative prob ≥ p. p=0.9. (4) Temperature: divide
logits by T. T<1 sharper, T>1 flatter. (5) Beam
search: maintain B beams.

KV cache: en autoregressive generation, cache K, V
previos. Memoria O(layers × seq_len × d_kv). GQA reduce
4-8x. vLLM paged attention optimiza.

Padded generation: left-pad para batch inference. mask
en attention.

nanoGPT (Karpathy 2022): 300 líneas, GPT-2 small
reproducible. Entrenar en 1 GPU ~30 min con Shakespeare.

Llama 3 (Meta 2024): 8B-405B params, 128K vocab, RoPE,
GQA, SwiGLU. 15.6T tokens training. SOTA en coding,
math, reasoning.

Tamaños prácticos: 100M (toy), 1B (research), 7B
(production), 70B (frontier), 405B+ (rare).

## Lecturas recomendadas

- "nanoGPT" (Karpathy 2022)
- "GPT-2" (Radford 2019)
- "Llama 3" (Meta 2024)
- "Top-p" (Holtzman 2020)

---

> 📚 **Adaptación al español** de la lección
> "[35-gpt-model-assembly]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
