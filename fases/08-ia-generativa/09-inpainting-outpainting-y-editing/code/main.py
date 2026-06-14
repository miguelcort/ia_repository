"""
Lección: 09-inpainting-outpainting-y-editing
Fase: 08
Prerrequisitos: 08/07-difusion-latente-stable-diffusion,
                08/08-controlnet-y-lora-condicionamiento
Fuentes:
- Lugmayr et al. (2022). RePaint: Inpainting using DDPMs
  (arXiv:2201.09865)
- Meng et al. (2022). SDEdit (arXiv:2108.01073)
- Brooks et al. (2023). InstructPix2Pix (arXiv:2211.09800)
- Hertz et al. (2022). Prompt-to-Prompt (arXiv:2208.01626)

Inpainter DDPM de juguete en 5-D. Enmascaramos 2 de 5
dimensiones, inyectamos la versión forward-noisy de las no
enmascaradas en cada paso, regeneramos solo las enmascaradas.
"""
from __future__ import annotations

import math
import random


def sin_embed(t: int, T: int, dim: int = 8) -> list[float]:
    """Embedding sinusoidal del timestep."""
    out: list[float] = []
    half = dim // 2
    for i in range(half):
        freq = 1.0 / (10000 ** (i / max(half - 1, 1)))
        out.append(math.sin(t * freq))
        out.append(math.cos(t * freq))
    return out[:dim]


def tanh(v: list[float]) -> list[float]:
    return [math.tanh(x) for x in v]


def tanh_grad(h: list[float]) -> list[float]:
    return [1 - x * x for x in h]


def matmul(W: list[list[float]], x: list[float]) -> list[float]:
    return [sum(w * xi for w, xi in zip(row, x)) for row in W]


def add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def randn_matrix(
    rows: int, cols: int, rng: random.Random, scale: float = 0.3
) -> list[list[float]]:
    return [[rng.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]


def init_net(
    x_dim: int, t_dim: int, hidden: int, rng: random.Random
) -> dict[str, list]:
    return {
        "W1": randn_matrix(hidden, x_dim + t_dim, rng),
        "b1": [0.0] * hidden,
        "W2": randn_matrix(hidden, hidden, rng),
        "b2": [0.0] * hidden,
        "W3": randn_matrix(x_dim, hidden, rng),
        "b3": [0.0] * x_dim,
    }


def forward(
    x_t: list[float], t_emb: list[float], net: dict
) -> tuple[list[float], dict]:
    inp = list(x_t) + list(t_emb)
    pre1 = add(matmul(net["W1"], inp), net["b1"])
    h1 = tanh(pre1)
    pre2 = add(matmul(net["W2"], h1), net["b2"])
    h2 = tanh(pre2)
    out = add(matmul(net["W3"], h2), net["b3"])
    return out, {"inp": inp, "h1": h1, "h2": h2}


def backward(
    target: list[float], out: list[float], cache: dict, net: dict
) -> dict:
    grads: dict = {}
    for k, v in net.items():
        if isinstance(v[0], list):
            grads[k] = [[0.0] * len(v[0]) for _ in v]
        else:
            grads[k] = [0.0] * len(v)
    d_out = [2 * (a - b) for a, b in zip(out, target)]
    for i in range(len(d_out)):
        grads["b3"][i] += d_out[i]
        for j in range(len(cache["h2"])):
            grads["W3"][i][j] += d_out[i] * cache["h2"][j]
    d_h2 = [
        sum(net["W3"][i][j] * d_out[i] for i in range(len(d_out)))
        for j in range(len(cache["h2"]))
    ]
    d_pre2 = [
        d_h2[j] * tanh_grad(cache["h2"])[j] for j in range(len(cache["h2"]))
    ]
    for j in range(len(d_pre2)):
        grads["b2"][j] += d_pre2[j]
        for k in range(len(cache["h1"])):
            grads["W2"][j][k] += d_pre2[j] * cache["h1"][k]
    d_h1 = [
        sum(net["W2"][j][k] * d_pre2[j] for j in range(len(d_pre2)))
        for k in range(len(cache["h1"]))
    ]
    d_pre1 = [
        d_h1[j] * tanh_grad(cache["h1"])[j] for j in range(len(cache["h1"]))
    ]
    for j in range(len(d_pre1)):
        grads["b1"][j] += d_pre1[j]
        for k in range(len(cache["inp"])):
            grads["W1"][j][k] += d_pre1[j] * cache["inp"][k]
    return grads


def apply(net: dict, grads: dict, lr: float) -> None:
    for k, v in net.items():
        if isinstance(v[0], list):
            for i in range(len(v)):
                for j in range(len(v[i])):
                    v[i][j] -= lr * grads[k][i][j]
        else:
            for i in range(len(v)):
                v[i] -= lr * grads[k][i]


def make_schedule(T: int) -> tuple[list[float], list[float]]:
    betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
    alphas = [1 - b for b in betas]
    bars, cum = [], 1.0
    for a in alphas:
        cum *= a
        bars.append(cum)
    return alphas, bars


def sample_data(
    rng: random.Random, d: int = 5
) -> tuple[list[float], int]:
    cluster = rng.choice([0, 1])
    center = [-1.0 if cluster == 0 else 1.0] * d
    return [c + rng.gauss(0, 0.2) for c in center], cluster


def train(
    net: dict,
    alpha_bars: list[float],
    T: int,
    steps: int,
    lr: float,
    t_dim: int,
    d: int,
    rng: random.Random,
) -> None:
    for step in range(steps):
        x0, _ = sample_data(rng, d)
        t = rng.randrange(T)
        eps = [rng.gauss(0, 1) for _ in range(d)]
        a_bar = alpha_bars[t]
        x_t = [
            math.sqrt(a_bar) * x0[i] + math.sqrt(1 - a_bar) * eps[i]
            for i in range(d)
        ]
        t_emb = sin_embed(t, T, t_dim)
        out, cache = forward(x_t, t_emb, net)
        grads = backward(eps, out, cache, net)
        apply(net, grads, lr)


def sample_unconditional(
    net: dict,
    alphas: list[float],
    alpha_bars: list[float],
    T: int,
    t_dim: int,
    d: int,
    rng: random.Random,
) -> list[float]:
    x = [rng.gauss(0, 1) for _ in range(d)]
    for t in range(T - 1, -1, -1):
        t_emb = sin_embed(t, T, t_dim)
        eps_hat, _ = forward(x, t_emb, net)
        beta_t = 1 - alphas[t]
        mean = [
            (
                x[i]
                - beta_t
                / math.sqrt(1 - alpha_bars[t])
                * eps_hat[i]
            )
            / math.sqrt(alphas[t])
            for i in range(d)
        ]
        if t > 0:
            x = [
                mean[i] + math.sqrt(beta_t) * rng.gauss(0, 1)
                for i in range(d)
            ]
        else:
            x = mean
    return x


def inpaint(
    net: dict,
    alphas: list[float],
    alpha_bars: list[float],
    T: int,
    t_dim: int,
    d: int,
    clean: list[float],
    mask: list[bool],
    rng: random.Random,
) -> list[float]:
    """mask[i] == True significa que esa dimensión se regenera.
    Las dimensiones no enmascaradas se *pinean* a clean."""
    x = [rng.gauss(0, 1) for _ in range(d)]
    for t in range(T - 1, -1, -1):
        a_bar = alpha_bars[t]
        for i in range(d):
            if not mask[i]:
                x[i] = (
                    math.sqrt(a_bar) * clean[i]
                    + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
                )
        t_emb = sin_embed(t, T, t_dim)
        eps_hat, _ = forward(x, t_emb, net)
        beta_t = 1 - alphas[t]
        mean = [
            (
                x[i]
                - beta_t
                / math.sqrt(1 - alpha_bars[t])
                * eps_hat[i]
            )
            / math.sqrt(alphas[t])
            for i in range(d)
        ]
        if t > 0:
            x = [
                mean[i] + math.sqrt(beta_t) * rng.gauss(0, 1)
                for i in range(d)
            ]
        else:
            x = mean
    for i in range(d):
        if not mask[i]:
            x[i] = clean[i]
    return x


def main() -> int:
    rng = random.Random(5)
    T, t_dim, hidden, d = 40, 8, 32, 5
    alphas, alpha_bars = make_schedule(T)
    net = init_net(d, t_dim, hidden, rng)

    print("=== entrenamiento DDPM 5-D sobre mezcla de dos clústeres ===")
    train(
        net, alpha_bars, T, steps=5000, lr=0.01, t_dim=t_dim, d=d, rng=rng
    )

    print()
    print("=== inpainting: pinea dims 0-2, regenera dims 3-4 ===")
    for trial in range(5):
        clean, cluster = sample_data(rng, d)
        mask = [False, False, False, True, True]
        out = inpaint(
            net, alphas, alpha_bars, T, t_dim, d, clean, mask, rng
        )
        label = "clúster neg" if cluster == 0 else "clúster pos"
        print(
            f"  {label}: pineadas="
            f"{[f'{clean[i]:+.2f}' for i in range(3)]}  "
            f"rellenas={[f'{out[i]:+.2f}' for i in range(3, 5)]}"
        )

    print()
    print("=== outpainting (enmascara dims 0-1, pinea 2-4) ===")
    for trial in range(3):
        clean, cluster = sample_data(rng, d)
        mask = [True, True, False, False, False]
        out = inpaint(
            net, alphas, alpha_bars, T, t_dim, d, clean, mask, rng
        )
        print(
            f"  pineada cola=[{clean[2]:+.2f}, {clean[3]:+.2f}, {clean[4]:+.2f}]  "
            f"rellena cabeza=[{out[0]:+.2f}, {out[1]:+.2f}]"
        )

    print()
    print("conclusión: las dims rellenas matchean el signo del clúster")
    print("            de las dims pineadas. Por eso el inpainting luce")
    print("            coherente con su entorno.")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
