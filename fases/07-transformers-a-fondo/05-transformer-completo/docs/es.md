# 05 — Transformer completo

> El transformer canónico (Vaswani et al., 2017) tiene encoder + decoder con self-attention, cross-attention, y feed-forward layers. BERT usa solo el encoder; GPT solo el decoder; T5 ambos.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-self-attention-desde-cero
**Tiempo estimado:** ~60 minutos

## Objetivos de aprendizaje

- Implementar el transformer encoder-decoder completo
  desde cero.
- Entender pre-norm vs post-norm.
- Diagnosticar los family trees: encoder-only,
  decoder-only, encoder-decoder.
- Conocer las variantes modernas: RoBERTa, ALBERT,
  DeBERTa, T5.

## El problema

El paper "Attention Is All You Need" (Vaswani et al.,
2017) introdujo la arquitectura transformer canónica
para traducción automática. Desde entonces, la familia
ha crecido: encoder-only (BERT), decoder-only (GPT),
encoder-decoder (T5). La lección implementa el
transformer canónico y mapea las variantes modernas.

## El concepto

**Transformer encoder-decoder (canónico).**

- **Encoder:** N=6 bloques, cada uno con multi-head
  self-attention + feed-forward. Skip connections +
  layer norm. Sin masking (atención bidireccional).
- **Decoder:** N=6 bloques, cada uno con masked self-
  attention (causal) + cross-attention al encoder +
  feed-forward.
- **Embeddings:** token + positional (sinusoidal en el
  paper original, aprendido en BERT/GPT).
- **Output:** linear projection al vocabulario + softmax.

**Pre-norm vs post-norm.**

- **Post-norm (original):** `LayerNorm(x + Sublayer(x))`.
  Más difícil de entrenar con muchas capas.
- **Pre-norm (moderno):** `x + Sublayer(LayerNorm(x))`. Más
  estable, usado en GPT, LLaMA, Mistral. Default en
  Hugging Face.

**Family tree.**

- **Encoder-only:** BERT, RoBERTa, DeBERTa, ELECTRA.
  Para clasificación, NER, embedding extraction.
  Bidireccional.
- **Decoder-only:** GPT-1/2/3/4, LLaMA, Mistral, Qwen.
  Para generación de texto. Autoregresivo (causal).
- **Encoder-decoder:** T5, BART, FLAN-T5, mBART. Para
  seq2seq (traducción, resumen, QA).
- **Mixture of Experts (MoE):** Mixtral, DeepSeek-V3.
  Reemplaza FFN con expertos sparse.

**Variantes modernas.**

- **RoBERTa (Liu et al., 2019):** mejor pre-entrenamiento
  de BERT (más datos, sin NSP, más pasos).
- **ALBERT (Lan et al., 2019):** parameter sharing entre
  capas. Modelos más pequeños.
- **DeBERTa (He et al., 2020):** disentangled
  attention. SOTA en NLI y clasificación.
- **T5 (Raffel et al., 2020):** text-to-text. Toda
  tarea reformulada como input → output.
- **ELECTRA (Clark et al., 2020):** replaced token
  detection. Más eficiente en pre-entrenamiento.

**Cuándo usar cada familia.**

- **Clasificación, NER, embeddings:** encoder-only
  (BERT, RoBERTa, DeBERTa).
- **Generación, chat, code:** decoder-only (GPT,
  LLaMA, Qwen).
- **Traducción, resumen, QA con contexto largo:**
  encoder-decoder (T5, BART, FLAN-T5).

**Trampas.**

- **Encoder-only para generación:** no aprende a generar
  token por token. Usa solo decoder o encoder-decoder.
- **Decoder-only para embeddings:** sin bidireccionalidad.
  Usa last hidden state o fine-tunea con pooling.
- **Sin pre-entrenamiento:** transformers sin
  pre-entrenamiento son inferiores a RNN/CNN. Siempre
  pre-entrenar con MLM, CLM, o seq2seq.

## Constrúyelo

```python
import numpy as np


def layer_norm(x, eps=1e-6):
    """Layer normalization: normaliza sobre la última dimensión."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def feed_forward(x, W1, b1, W2, b2):
    """FFN del transformer: dos lineales con ReLU."""
    return np.maximum(0, x @ W1 + b1) @ W2 + b2


def encoder_block(x, W_qkv, W_o, W_ff1, b_ff1, W_ff2, b_ff2,
                n_heads):
    """Un bloque del encoder (pre-norm)."""
    # Pre-norm + self-attention
    x_norm = layer_norm(x)
    attn_out = multi_head_attention(x_norm, W_qkv[:, :d],
                                    W_qkv[:, d:2*d],
                                    W_qkv[:, 2*d:3*d], W_o, n_heads)
    x = x + attn_out
    # Pre-norm + FFN
    x_norm = layer_norm(x)
    ffn_out = feed_forward(x_norm, W_ff1, b_ff1, W_ff2, b_ff2)
    x = x + ffn_out
    return x
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-transformer-family
fase: 07
leccion: 05
---

Eres un asistente que ayuda a elegir arquitectura de
transformer. Recibirás la tarea y los recursos. Tu trabajo:

1. Para clasificación, NER, embeddings: encoder-only
   (DeBERTa-v3, RoBERTa).
2. Para generación, chat, code: decoder-only (LLaMA,
   Qwen, Mistral).
3. Para seq2seq, traducción, QA: encoder-decoder
   (T5, FLAN-T5).
4. Para recursos limitados: DistilBERT, MobileBERT.
5. Para eficiencia: FlashAttention, ALBERT.
6. Pre-entrenado: usar siempre modelos pre-entrenados,
   entrenar desde cero solo si tienes mucho cómputo.
7. Para billion+ parámetros: usar MoE (Mixtral,
   DeepSeek-V3) para降低成本.
```

## Ejercicios

1. **Encoder block**: implementa y visualiza los
   gradientes.
2. **Pre-norm vs post-norm**: compara estabilidad de
   entrenamiento.
3. **Desafío**: implementa un mini-GPT (decoder-only)
   y entrena en TinyStories.

## Lecturas recomendadas

- *Attention Is All You Need* — Vaswani et al., 2017.
- *BERT* — Devlin et al., 2018.
- *RoBERTa* — Liu et al., 2019.
- *T5* — Raffel et al., 2020.
- HuggingFace Transformers: <https://huggingface.co/docs/transformers>.

---

> 📚 **Adaptación al español** de la lección "[Full Transformer]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
