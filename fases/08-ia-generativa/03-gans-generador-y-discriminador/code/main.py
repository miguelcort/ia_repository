"""
Lección: 03-gans-generador-y-discriminador
Fase: 08
GAN: G(z) -> x fake, D(x) -> real/fake. Entrenamiento adversarial.
"""
from __future__ import annotations
import sys
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generator_forward(z, W1, b1, W2, b2):
    """z: (n, latent_dim) -> (n, output_dim)."""
    h = np.maximum(0, z @ W1 + b1)  # ReLU
    return _sigmoid(h @ W2 + b2)


def discriminator_forward(x, W1, b1, W2, b2):
    """x: (n, dim) -> (n, 1) logit."""
    h = np.maximum(0, x @ W1 + b1)  # Leaky ReLU
    return h @ W2 + b2


def gan_d_loss(d_real_logits, d_fake_logits, mode="non-saturating"):
    """Discriminator loss. Mode original: minimize -log(D(x_real)) - log(1-D(x_fake))."""
    eps = 1e-9
    if mode == "vanilla":
        # Original GAN: D wants to maximize log(D(x)) + log(1 - D(G(z)))
        return float(-np.log(_sigmoid(d_real_logits) + eps).mean()
                     - np.log(1 - _sigmoid(d_fake_logits) + eps).mean())
    elif mode == "non-saturating":
        # Wasserstein-style: D wants D(real) - D(fake) large
        return float(-d_real_logits.mean() + d_fake_logits.mean())
    return 0.0


def gan_g_loss(d_fake_logits, mode="non-saturating"):
    """Generator loss. Saturating: -log(D(G(z))). Non-saturating: -D(G(z))."""
    eps = 1e-9
    if mode == "vanilla":
        return float(-np.log(_sigmoid(d_fake_logits) + eps).mean())
    elif mode == "non-saturating":
        return float(-d_fake_logits.mean())
    return 0.0


def wasserstein_loss(d_real, d_fake):
    """WGAN: critic loss = D(real) - D(fake), G loss = -D(fake).
    Critic debe estar 1-Lipschitz (weight clipping o gradient penalty).
    """
    return float(d_fake.mean() - d_real.mean())


def gradient_penalty(discriminator_fn, real, fake, eps=1e-9):
    """WGAN-GP: ||grad_x D(x_hat)||^2 - 1, x_hat = eps*real + (1-eps)*fake.
    Asegura 1-Lipschitz sin weight clipping.
    """
    # Interpolar
    rng = np.random.default_rng(0)
    alpha = rng.uniform(size=(real.shape[0], 1))
    x_hat = alpha * real + (1 - alpha) * fake
    # Computar logits en x_hat (numericamente usamos finite differences)
    eps_fd = 1e-4
    grad = np.zeros_like(x_hat)
    for i in range(x_hat.shape[1]):
        x_plus = x_hat.copy()
        x_plus[:, i] += eps_fd
        x_minus = x_hat.copy()
        x_minus[:, i] -= eps_fd
        d_plus = discriminator_fn(x_plus)
        d_minus = discriminator_fn(x_minus)
        grad[:, i] = (d_plus - d_minus) / (2 * eps_fd)
    # ||grad||^2
    return float((grad ** 2).sum(axis=1).mean())


def init_weights(in_dim, out_dim, seed=0):
    rng = np.random.default_rng(seed)
    s = np.sqrt(2.0 / in_dim)  # He init
    return s * rng.standard_normal((in_dim, out_dim))


def main() -> int:
    latent, hidden, dim = 8, 16, 4
    W_g1 = init_weights(latent, hidden, seed=0)
    b_g1 = np.zeros(hidden)
    W_g2 = init_weights(hidden, dim, seed=1)
    b_g2 = np.zeros(dim)
    W_d1 = init_weights(dim, hidden, seed=2)
    b_d1 = np.zeros(hidden)
    W_d2 = init_weights(hidden, 1, seed=3)
    b_d2 = np.zeros(1)
    z = np.random.default_rng(0).standard_normal((3, latent))
    fake = generator_forward(z, W_g1, b_g1, W_g2, b_g2)
    print(f"Generated shape: {fake.shape}, range [{fake.min():.3f}, {fake.max():.3f}]")
    real = np.random.default_rng(10).standard_normal((3, dim))
    d_real = discriminator_forward(real, W_d1, b_d1, W_d2, b_d2)
    d_fake = discriminator_forward(fake, W_d1, b_d1, W_d2, b_d2)
    print(f"D(real)={d_real.mean():.3f}, D(fake)={d_fake.mean():.3f}")
    d_loss, g_loss = (gan_d_loss(d_real, d_fake, "non-saturating"),
                      gan_g_loss(d_fake, "non-saturating"))
    print(f"D_loss={d_loss:.3f}, G_loss={g_loss:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())