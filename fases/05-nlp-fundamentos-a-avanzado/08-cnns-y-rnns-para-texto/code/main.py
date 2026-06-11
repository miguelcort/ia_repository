"""
Lección: 08-cnns-y-rnns-para-texto
Fase: 05
Prerrequisitos: 07-etiquetado-pos-y-parsing
"""
from __future__ import annotations
import sys
import numpy as np


def text_cnn_kernel(x_emb, kernels, bias=0.0):
    """CNN 1D para texto. x_emb: (T, D). kernels: lista de (k, D).
    Cada kernel produce un feature map de tamano T - k + 1.
    Aplica max-pooling y concatena.
    """
    T, D = x_emb.shape
    mapas = []
    for k in kernels:
        k_arr = np.asarray(k)
        k_size = k_arr.shape[0]
        n_out = T - k_size + 1
        feat = np.zeros(n_out)
        for i in range(n_out):
            ventana = x_emb[i:i + k_size]
            feat[i] = float((ventana * k_arr).sum()) + bias
        mapas.append(feat.max())
    return np.array(mapas)


def rnn_cell_simple(x_t, h_prev, W_hh, W_xh, b_h, activacion="tanh"):
    """Celda RNN simple. h_t = tanh(W_hh h_{t-1} + W_xh x_t + b_h)."""
    h_pre = W_hh @ h_prev + W_xh @ x_t + b_h
    if activacion == "tanh":
        return np.tanh(h_pre)
    elif activacion == "relu":
        return np.maximum(0, h_pre)
    return h_pre


def bidirectional_rnn(x, h0, W_hh_f, W_xh_f, W_hh_b, W_xh_b, b_f, b_b):
    """BiRNN: forward + backward. Devuelve concatenacion de h_f y h_b en cada paso.
    x: (T, D). h0: (H,). W_*_f, W_*_b: (H, H) y (H, D). b_*: (H,)."""
    T, D = x.shape
    H = h0.shape[0]
    h_f = h0.copy()
    h_b = h0.copy()
    forwards = []
    backwards = []
    for t in range(T):
        h_f = rnn_cell_simple(x[t], h_f, W_hh_f, W_xh_f, b_f)
        forwards.append(h_f)
    for t in range(T - 1, -1, -1):
        h_b = rnn_cell_simple(x[t], h_b, W_hh_b, W_xh_b, b_b)
        backwards.insert(0, h_b)
    # Concat forward y backward
    bi = np.zeros((T, 2 * H))
    for t in range(T):
        bi[t] = np.concatenate([forwards[t], backwards[t]])
    return bi


def main() -> int:
    # CNN
    T, D = 10, 8
    x = np.random.default_rng(0).normal(size=(T, D))
    kernels = [np.random.default_rng(1).normal(size=(3, D)) for _ in range(3)]
    feats = text_cnn_kernel(x, kernels)
    print(f"TextCNN features: {feats.shape}")
    # RNN
    H = 16
    rng = np.random.default_rng(0)
    W_hh = rng.normal(scale=0.1, size=(H, H))
    W_xh = rng.normal(scale=0.1, size=(H, D))
    b = np.zeros(H)
    h = np.zeros(H)
    hiddens = [h]
    for t in range(T):
        h = rnn_cell_simple(x[t], h, W_hh, W_xh, b)
        hiddens.append(h)
    print(f"Final h shape: {h.shape}, total pasos: {len(hiddens) - 1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())