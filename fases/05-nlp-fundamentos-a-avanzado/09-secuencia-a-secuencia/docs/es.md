# Secuencia a secuencia (seq2seq)

> Encoder-decoder: el encoder lee el input, el decoder genera la salida condicionado al contexto. La base de MT, summarization, QA. Hoy: T5, BART, mT5, NLLB. Attention resolvio el information bottleneck del seq2seq LSTM original.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 08-cnns-y-rnns-para-texto
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar encoder RNN.
- Implementar decoder step con softmax.
- Encadenar encode -> decode.
- Diagnosticar information bottleneck y atencion.

## Constrúyelo

```python
def seq2seq(x, h0, W_hh, W_xh, W_hy, b_h, b_y, vocab, max_len=10):
    h = encoder_rnn(x, h0, W_hh, W_xh, b_h)
    y_prev = np.zeros(len(vocab))
    y_prev[0] = 1.0  # <sos>
    output = []
    for _ in range(max_len):
        h, _, idx, tok = decoder_rnn_step(y_prev, h, W_hh, W_xh, W_hy, b_h, b_y, vocab)
        output.append(tok)
        if tok == "<eos>": break
        y_prev = np.zeros(len(vocab))
        y_prev[idx] = 1.0
    return output
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-seq2seq
fase: 05
leccion: 09
---

1. MT: NLLB-200, mBART.
2. Summarization: BART, T5.
3. QA extractiva: T5, RoBERTa.
4. Captioning: BLIP-2, LLaVA.
5. Generation abierta: Llama, Mistral.
6. Beam 4-8 con length penalty.
```

## Ejercicios

1. **Beam search**: implementar beam search con k=4.
2. **Attention basica**: implementar attention
   encoder-decoder.
3. **Desafio**: fine-tunear T5-small en summarization de
   noticias en espanol, alcanzar ROUGE-L > 0.30.

## Lecturas recomendaciones

- "Sequence to Sequence Learning with Neural Networks"
  (Sutskever et al., 2014)
- "BART" (Lewis et al., 2020)
- "T5" (Raffel et al., 2020)

---

> 📚 **Adaptación al español** de la lección "[Sequence to Sequence]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).