# 02 — Self-attention desde cero

> Self-attention es la operación central de los transformers. Cada token atiende a todos los demás con pesos aprendibles. La lección cubre scaled dot-product attention, multi-head, y masking.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 01-por-que-transformers
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Implementar scaled dot-product attention desde cero.
- Implementar multi-head attention.
- Aplicar causal masking para autoregresive language
  models.
- Diagnosticar la complejidad O(n²) y cuándo usar
  variantes eficientes.

## El problema

Self-attention es la operación que hace a los transformers
tan poderosos. Cada token en una secuencia produce tres
vectores: Query (Q), Key (K), Value (V). La atención se
computa como una softmax sobre scores Q·K^T,
ponderando V. Multi-head permite aprender múltiples
relaciones en paralelo. La lección implementa todo
desde cero.

## El concepto

**Scaled dot-product attention (Vaswani et al., 2017).**

```text
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

Q, K, V son matrices `(seq_len, d_k)`. La división por
`sqrt(d_k)` evita que los scores se vuelvan demasiado
grandes, lo que saturaría la softmax.

**Multi-head attention.** H cabezas en paralelo:

```text
MultiHead(Q, K, V) = Concat(head_1, ..., head_H) W_O
head_i = Attention(Q W_Q_i, K W_K_i, V W_V_i)
```

Cada cabeza aprende diferentes relaciones: sintácticas,
semánticas, coreferencia. Típico H=8 con d_k = d_model /
H = 64.

**Causal masking.** Para autoregresive LM (GPT), cada
token solo puede atender a tokens pasados. Aplicar una
máscara triangular a los scores: `score[i, j] = -inf
si j > i`. Softmax(0) = 0, así que las posiciones futuras
reciben peso 0.

**Cross-attention.** En el decoder, self-attention sobre
el target + cross-attention sobre la salida del encoder.
Misma fórmula, pero Q viene del decoder y K, V del
encoder.

**Complejidad.**

- **Memoria:** O(n²) por la matriz de attention.
- **Compute:** O(n² d).
- **FlashAttention (Dao et al., 2022):** memory-efficient
  attention via tiling. Reduce memoria a O(n) sin
  cambiar la math.

**Cuándo usar self-attention.**

- **SOTA en NLP:** BERT, GPT, T5.
- **Visión:** ViT, Swin.
- **Audio:** Whisper, Qwen-Audio.
- **Multimodal:** CLIP, LLaVA, Qwen-VL.

**Trampas.**

- **Attention sin escala:** los scores explotan, la
  softmax satura. Siempre dividir por sqrt(d_k).
- **Sin causal mask en LM:** el modelo ve el futuro y
  memoriza en vez de predecir.
- **Memoria O(n²):** para secuencias > 4k tokens, usar
  Longformer, FlashAttention, o chunking.

## Constrúyelo

```python
import numpy as np


def softmax(x, axis=-1):
    """Softmax numéricamente estable."""
    x_max = x.max(axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled dot-product attention.
    Q, K, V: (seq_len, d_k). mask: (seq_len, seq_len), 1=attend, 0=mask.
    Devuelve (output, attention_weights)."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = scores + (1 - mask) * -1e9
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


def multi_head_attention(X, W_q, W_k, W_v, W_o, n_heads,
                        causal=False):
    """Multi-head self-attention.
    X: (seq_len, d_model). W_q/k/v/o: (d_model, d_model).
    n_heads: número de cabezas.
    Devuelve (seq_len, d_model)."""
    seq_len, d_model = X.shape
    d_k = d_model // n_heads
    # Proyecciones
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    # Split en cabezas: (n_heads, seq_len, d_k)
    Q = Q.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    K = K.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    V = V.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    # Máscara causal
    if causal:
        mask = np.tril(np.ones((seq_len, seq_len)))
        # Expandir a (n_heads, seq_len, seq_len)
        mask = np.broadcast_to(mask, (n_heads, seq_len, seq_len))
    else:
        mask = None
    # Atención por cabeza
    out_heads = []
    for h in range(n_heads):
        out, _ = scaled_dot_product_attention(Q[h], K[h], V[h],
                                                mask=mask[h] if mask is not None else None)
        out_heads.append(out)
    # Concatenar cabezas
    out = np.stack(out_heads, axis=1).reshape(seq_len, d_model)
    return out @ W_o
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-attention
fase: 07
leccion: 02
---

Eres un asistente que ayuda con self-attention. Reci-
birás la arquitectura. Tu trabajo:

1. Para producción: PyTorch nn.MultiheadAttention o
   transformers library.
2. Para memoria limitada: FlashAttention.
3. Para secuencias largas: Longformer, BigBird, o
   chunking.
4. Para LMs: usar causal mask.
5. Para encoder-only (BERT): no causal mask, attention
   bidireccional.
6. Para cross-attention: Q del decoder, K/V del encoder.
7. Advertir contra attention sin escala.
```

## Ejercicios

1. **Self-attention**: implementa y aplica a una
   secuencia de embeddings.
2. **Causal mask**: visualiza la máscara triangular y
   verifica que las posiciones futuras tienen peso 0.
3. **Desafío**: implementa FlashAttention simplificado
   con tiling.

## Lecturas recomendadas

- *Attention Is All You Need* — Vaswani et al., 2017.
- *FlashAttention* — Dao et al., 2022.
- *The Illustrated Transformer* — Jay Alammar.
- *Formal Limitations of Transformers* — Hahn, 2020.

---

> 📚 **Adaptación al español** de la lección "[Self-Attention from Scratch]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
