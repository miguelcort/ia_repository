"""
Lección: 02-reward-hacking-goodhart
Fase: 18
Prerrequisitos: 18/01-instruction-following-alignment-signal,
                10-llms-desde-cero/07-rlhf
Fuentes:
- Gao, Schulman, Hilton — Scaling Laws for Reward Model Overoptimization
  (ICML 2023)
- Manheim & Garrabrant — Categorizing Variants of Goodhart's Law
  (arXiv:1803.04585)
- Catastrophic Goodhart (OpenReview UXuBzWoZGK)
"""
from __future__ import annotations

import math
import random


def gold_reward(weights: list[float], features: list[float]) -> float:
    """Recompensa 'oro': producto punto verdadero."""
    return sum(w * f for w, f in zip(weights, features))


def fit_proxy_rm(
    weights: list[float],
    features_list: list[list[float]],
    noise_scale: float = 0.5,
    tail_df: float | None = None,
) -> list[float]:
    """Ajusta un RM 'proxy' por regresión lineal con ruido.
    Si tail_df está dado, usa Student-t(grados_de_libertad) en
    lugar de Gaussiana para el ruido (cola pesada)."""
    n_features = len(weights)
    XtX = [[0.0] * n_features for _ in range(n_features)]
    Xty = [0.0] * n_features
    for features in features_list:
        y = gold_reward(weights, features)
        if tail_df is not None:
            # Muestra una t-Student con cola pesada
            noise = _sample_student_t(tail_df) * noise_scale
        else:
            noise = random.gauss(0, noise_scale)
        y_noisy = y + noise
        for i in range(n_features):
            Xty[i] += features[i] * y_noisy
            for j in range(n_features):
                XtX[i][j] += features[i] * features[j]
    # Resuelve (X^T X) w = X^T y por Gauss-Jordan
    return _solve_linear(XtX, Xty)


def _sample_student_t(df: float) -> float:
    """Muestra de t-Student con df grados de libertad (aproximación
    simple con normal / sqrt(chi2/df))."""
    normal = random.gauss(0, 1)
    chi2 = sum(random.gauss(0, 1) ** 2 for _ in range(max(1, int(df))))
    return normal / math.sqrt(chi2 / df)


def _solve_linear(A: list[list[float]], b: list[float]) -> list[float]:
    """Resuelve A x = b por eliminación gaussiana (in-place)."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        # pivote
        pivot = M[i][i]
        if abs(pivot) < 1e-12:
            for k in range(i + 1, n):
                if abs(M[k][i]) > abs(pivot):
                    M[i], M[k] = M[k], M[i]
                    pivot = M[i][i]
                    break
        pivot = M[i][i]
        for j in range(n + 1):
            M[i][j] /= pivot
        for k in range(n):
            if k != i:
                factor = M[k][i]
                for j in range(n + 1):
                    M[k][j] -= factor * M[i][j]
    return [M[i][n] for i in range(n)]


def proxy_reward(proxy_weights: list[float], features: list[float]) -> float:
    return sum(w * f for w, f in zip(proxy_weights, features))


def kl_divergence_gaussians(
    mu1: list[float], sigma1: float, mu2: list[float], sigma2: float
) -> float:
    """KL entre dos Gaussianas isotrópicas multidimensionales."""
    d = len(mu1)
    var_ratio = sigma1**2 / sigma2**2
    mean_diff_sq = sum((m1 - m2) ** 2 for m1, m2 in zip(mu1, mu2))
    return 0.5 * (
        d * (var_ratio - 1.0)
        - math.log(var_ratio)
        + mean_diff_sq / (sigma2**2)
    )


def hill_climb(
    initial_features: list[float],
    proxy_weights: list[float],
    initial_weights: list[float],
    initial_sigma: float = 0.3,
    n_steps: int = 60,
    step_size: float = 0.05,
    sigma_decay: float = 0.97,
    beta: float = 0.1,
) -> list[tuple[float, float, list[float]]]:
    """Hill-climbing con KL penalty. Devuelve la trayectoria
    (KL, proxy, features) en cada paso."""
    trajectory: list[tuple[float, float, list[float]]] = []
    features = list(initial_features)
    sigma = initial_sigma
    for _ in range(n_steps):
        # KL entre la Gaussiana actual (media=features, sigma) y la inicial
        kl = kl_divergence_gaussians(
            features, sigma, initial_features, initial_sigma
        )
        proxy = proxy_reward(proxy_weights, features)
        trajectory.append((math.sqrt(max(kl, 0.0)), proxy, list(features)))
        # dirección del gradiente proxy
        grad = list(proxy_weights)
        # update: gradiente ascent - beta * KL grad
        kl_grad = [(features[i] - initial_features[i]) / (initial_sigma**2)
                   for i in range(len(features))]
        for i in range(len(features)):
            features[i] += step_size * (grad[i] - beta * kl_grad[i])
        sigma *= sigma_decay
    return trajectory


def main() -> int:
    """Demo de la curva de sobre-optimización."""
    random.seed(0)

    print("=== 02-reward-hacking-goodhart ===\n")

    # Pesos verdaderos (gold) y dimensión
    true_weights = [1.0, -0.5, 0.3]
    dim = len(true_weights)

    # Genera datos: features uniformes
    def sample_feature():
        return [random.uniform(-1, 1) for _ in range(dim)]

    for n_samples, noise, df in [
        (100, 0.5, None),
        (300, 0.5, None),
        (1000, 0.5, None),
    ]:
        feats = [sample_feature() for _ in range(n_samples)]
        proxy_w = fit_proxy_rm(true_weights, feats, noise, tail_df=df)
        # hill-climb contra el proxy
        initial = [0.0] * dim
        traj = hill_climb(
            initial_features=initial,
            proxy_weights=proxy_w,
            initial_weights=true_weights,
            n_steps=60,
            beta=0.1,
        )
        # reporte: KL y oro en cada paso
        print(f"Proxy entrenado con n={n_samples}:")
        print("  KL_sqrt   proxy   gold")
        for kl, proxy, features in traj[::15]:
            gold = gold_reward(true_weights, features)
            print(f"  {kl:.3f}  {proxy:+.3f}  {gold:+.3f}")
        print()

    # Demostración de Catastrophic Goodhart con cola pesada
    print("Catastrophic Goodhart (Student-t df=2 vs Gaussiana):")
    feats = [sample_feature() for _ in range(1000)]
    for label, df in [("gaussiana", None), ("Student-t df=2", 2.0)]:
        proxy_w = fit_proxy_rm(true_weights, feats, 0.5, tail_df=df)
        initial = [0.0] * dim
        traj = hill_climb(
            initial_features=initial,
            proxy_weights=proxy_w,
            initial_weights=true_weights,
            n_steps=60,
            beta=0.1,
        )
        kl, proxy, features = traj[-1]
        gold = gold_reward(true_weights, features)
        print(
            f"  {label:>16}:  KL_sqrt={kl:.3f}  "
            f"proxy={proxy:+.3f}  gold={gold:+.3f}"
        )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
