"""
Lección: 13-leyes-de-escalado
Fase: 07
Chinchilla scaling laws: compute-optimal N y D. Loss = a * N^-alpha + b * D^-beta + e.
"""
from __future__ import annotations
import sys
import numpy as np


def chinchilla_loss(N, D, a=406.4, b=410.7, alpha=0.34, beta=0.28, e=1.69):
    """Loss predicha por Chinchilla: L = a * N^-alpha + b * D^-beta + e.
    N: params (no embeddings), D: tokens.
    """
    return a * (N ** -alpha) + b * (D ** -beta) + e


def compute_optimal(N_or_D, C, alpha=0.34, beta=0.28):
    """Dado budget C = 6 * N * D, calcular N* y D* optimos.
    Asume: N optimo ~ (alpha / (alpha + beta)) * C^(1/(alpha+beta+1)) * factor
    """
    # Simplificacion: ratio optimo N/D ~ 1 (Gopher, Chinchilla)
    # N = D = sqrt(C/6) cuando alpha = beta = 0.34 ~= 0.28, ratio optimo ~1.4
    # Aqui resolvemos aprox: alpha/beta * D = N (Chinchilla paper)
    ratio = alpha / beta  # 0.34/0.28 ~ 1.21
    # N = ratio * D, C = 6 * N * D = 6 * ratio * D^2
    # D = sqrt(C / (6 * ratio))
    D = np.sqrt(C / (6 * ratio))
    N = ratio * D
    return N, D


def predict_loss_for_budget(C):
    """Predice loss final dado compute budget C (FLOPs)."""
    N, D = compute_optimal(None, C)
    return chinchilla_loss(N, D)


def scale_gpt(n_params, n_tokens):
    """GPT-3: 175B params, 300B tokens. Loss tipico: 2.0-2.5.
    Para overtraining: misma params, +tokens -> loss menor.
    """
    # 6ND es el compute
    C = 6 * n_params * n_tokens
    return chinchilla_loss(n_params, n_tokens), C


def main() -> int:
    # Chinchilla 70B: 1.4T tokens
    loss_70b_1_4t = chinchilla_loss(N=70e9, D=1.4e12)
    print(f"Chinchilla 70B / 1.4T: loss = {loss_70b_1_4t:.2f}")
    # Gopher 280B: 380B tokens (suboptimo)
    loss_gopher = chinchilla_loss(N=280e9, D=380e9)
    print(f"Gopher 280B / 380B: loss = {loss_gopher:.2f}")
    # Compute-optimal 280B
    N_opt, D_opt = compute_optimal(None, 6 * 280e9 * 380e9)
    loss_opt = chinchilla_loss(N_opt, D_opt)
    print(f"Compute-optimal 280B: N={N_opt/1e9:.1f}B, D={D_opt/1e9:.1f}B, loss={loss_opt:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())