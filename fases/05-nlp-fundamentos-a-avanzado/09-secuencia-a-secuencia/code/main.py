"""
Lección: 09-secuencia-a-secuencia
Fase: 05
Prerrequisitos: 08-cnns-y-rnns-para-texto
"""
from __future__ import annotations
import sys
import numpy as np


def rnn_cell_simple(x_t, h_prev, W_hh, W_xh, b_h):
    """Celda RNN simple."""
    return np.tanh(W_hh @ h_prev + W_xh @ x_t + b_h)


def encoder_rnn(x, h0, W_hh, W_xh, b_h):
    """Encoder: procesa secuencia, devuelve h_final (contexto)."""
    T = len(x)
    h = h0.copy()
    for t in range(T):
        h = rnn_cell_simple(x[t], h, W_hh, W_xh, b_h)
    return h


def decoder_rnn_step(y_prev, h_prev, W_hh, W_xh, W_hy, b_h, b_y, vocab):
    """Decoder step: y_prev (V,) se proyecta a D usando una matriz de embedding.
    h_t = RNN(y_emb, h_prev), y_t = softmax(W_hy h_t).
    """
    D = W_xh.shape[1]
    V = y_prev.shape[0]
    # Embedding simple: matriz de pesos aleatoria fija (D, V) - para mantener dimensiones.
    # En la practica, esta seria la matriz de embedding aprendida.
    rng = np.random.default_rng(V)
    W_embed = rng.normal(scale=0.1, size=(D, V))
    y_emb = W_embed @ y_prev  # (D,)
    h = rnn_cell_simple(y_emb, h_prev, W_hh, W_xh, b_h)
    logits = W_hy @ h + b_y
    z = logits - logits.max()
    exp = np.exp(z)
    probs = exp / exp.sum()
    idx = int(probs.argmax())
    idx = min(idx, len(vocab) - 1)  # clamp para evitar OOB
    return h, probs, idx, vocab[idx]


def seq2seq(x, h0, W_hh, W_xh, W_hy, b_h, b_y, vocab, max_len=10):
    """Pipeline completo: encode -> decode hasta <eos> o max_len."""
    h = encoder_rnn(x, h0, W_hh, W_xh, b_h)
    y_prev = np.zeros(len(vocab))
    y_prev[0] = 1.0
    output = []
    for _ in range(max_len):
        h, _, idx, tok = decoder_rnn_step(y_prev, h, W_hh, W_xh, W_hy, b_h, b_y, vocab)
        output.append(tok)
        if tok == "<eos>":
            break
        y_prev = np.zeros(len(vocab))
        y_prev[idx] = 1.0
    return output


def main() -> int:
    V, H, D = 100, 16, 8
    rng = np.random.default_rng(0)
    W_hh = rng.normal(scale=0.1, size=(H, H))
    W_xh = rng.normal(scale=0.1, size=(H, D))
    W_hy = rng.normal(scale=0.1, size=(V, H))
    b_h = np.zeros(H)
    b_y = np.zeros(V)
    h0 = np.zeros(H)
    # Encoder
    x = [rng.normal(size=D) for _ in range(5)]
    h = encoder_rnn(x, h0, W_hh, W_xh, b_h)
    print(f"Encoder h_final shape: {h.shape}")
    # Seq2seq
    vocab = ["<sos>", "hola", "mundo", "<eos>"]
    out = seq2seq(x, h0, W_hh, W_xh, W_hy, b_h, b_y, vocab, max_len=5)
    print(f"Output: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())