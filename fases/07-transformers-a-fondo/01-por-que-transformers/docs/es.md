# Por qué transformers

> Self-attention O(1) path length vs RNN O(n). Paralelizable, escala, domina NLP/vision/audio. Limitacion: O(n^2) memoria. Alternativas: Mamba, RWKV, Hyena (linear/state-space) para long context, edge, streaming.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-nucleo-deep-learning/10-mini-framework
**Tiempo estimado:** ~25 minutos

## Objetivos de aprendizaje

- Calcular path length por arquitectura.
- Comparar complejidad computacional.
- Diagnosticar cuando transformer vs RNN.
- Diagnosticar inductive bias.

## Constrúyelo

```python
def path_length_analysis(seq_len, model="rnn"):
    if model == "rnn": return seq_len
    if model == "self-attention": return 1
    if model == "cnn": return int(np.ceil(np.log(seq_len) / np.log(3)))
    return -1
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-transformers-vs-rnn
fase: 07
leccion: 01
---

1. Default: Transformer.
2. Edge: DistilBERT, MobileBERT, Mamba.
3. Streaming/long: Mamba, RWKV, Hyena.
4. Datos pequenos: LR, SVM, XGBoost.
5. Flash Attention, KV cache, Mamba 2.
```

## Ejercicios

1. **Path length**: graficar path length vs seq_len para
   RNN, CNN, self-attention.
2. **Memory**: medir memoria O(n^2) para n=1K, 4K, 16K.
3. **Desafio**: comparar transformer, Mamba, y RNN en
   small text classification. Discutir tradeoffs.

## Lecturas recomendadas

- "Attention Is All You Need" (Vaswani et al., 2017)
- "On the Inductive Bias of Neural Networks" (Rahaman et al., 2019)
- "Mamba" (Gu & Dao, 2023)

---

> 📚 **Adaptación al español** de la lección "[Why Transformers]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).