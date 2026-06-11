# Mecanismo de atención

> La invencion que permitio transformers: cada token atiende a todos los demas con pesos aprendidos. Resolvio el information bottleneck del seq2seq. Multi-head + positional encodings + scaled dot-product = la receta de "Attention is All You Need".

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 09-secuencia-a-secuencia
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar scaled dot-product attention.
- Implementar attention Bahdanau y Luong.
- Diagnosticar multi-head attention.
- Diagnosticar positional encodings.

## Constrúyelo

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    pesos = softmax(scores, axis=-1)
    return pesos @ V, pesos
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
fase: 05
leccion: 10
---

1. Clasif: BERT (bidirectional).
2. Generation: GPT/Llama (causal).
3. Seq2seq: T5, BART (cross-attention).
4. Long context: Mistral sliding window, Longformer.
5. Multi-head 8-16, PE seno o RoPE, Flash Attention GPU.
```

## Ejercicios

1. **Multi-head**: implementar N cabezas en paralelo con
   concatenacion y proyeccion.
2. **Causal mask**: implementar mascara triangular para
   GPT-style.
3. **Desafio**: implementar self-attention completo con
   PE senos/cosenos y validar con shapes de BERT-base.

## Lecturas recomendadas

- "Neural Machine Translation by Jointly Learning to Align
  and Translate" (Bahdanau et al., 2015)
- "Effective Approaches to Attention-based Neural Machine
  Translation" (Luong et al., 2015)
- "Attention Is All You Need" (Vaswani et al., 2017)

---

> 📚 **Adaptación al español** de la lección "[Attention Mechanism]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).