# CNNs y RNNs para texto

> Antes de transformers, las arquitecturas dominantes. TextCNN (Kim 2014) para clasif rapida, BiLSTM para sequence labeling, encoder-decoder LSTM para MT. Hoy transformers dominaron, pero los conceptos son base.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07-etiquetado-pos-y-parsing
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar TextCNN 1D con max-pooling.
- Implementar RNN cell simple.
- Implementar BiRNN.
- Diagnosticar vanishing gradient y soluciones.

## Constrúyelo

```python
def rnn_cell_simple(x_t, h_prev, W_hh, W_xh, b_h):
    return np.tanh(W_hh @ h_prev + W_xh @ x_t + b_h)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-text-arch
fase: 05
leccion: 08
---

1. Clasif rapida: TextCNN, DistilBERT.
2. Sequence labeling: BiLSTM-CRF, BERT.
3. Max accuracy: BERT, RoBERTa.
4. Streaming/edge: RNN, Mamba, MobileBERT.
5. Seq2seq: encoder-decoder transformer.
```

## Ejercicios

1. **LSTM cell**: implementar las 3 compuertas (forget,
   input, output) y cell state.
2. **GRU**: implementar con 2 compuertas (reset, update).
3. **Desafio**: implementar BiLSTM-CRF para NER en
   CoNLL-2003 espanol.

## Lecturas recomendadas

- "Convolutional Neural Networks for Sentence Classification"
  (Kim, 2014) — TextCNN
- "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)
- "Empirical Evaluation of Gated Recurrent Neural Networks"
  (Chung et al., 2014) — GRU

---

> 📚 **Adaptación al español** de la lección "[CNNs and RNNs for Text]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).