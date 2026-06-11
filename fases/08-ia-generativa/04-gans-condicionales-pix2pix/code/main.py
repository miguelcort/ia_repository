"""
Lección: 04-gans-condicionales-pix2pix
Fase: 08
Conditional GAN: condicionar G y D en c (label, image, text). pix2pix: image-to-image.
"""
from __future__ import annotations
import sys
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def concat_condition(z, c, axis=-1):
    """Concatenar z y c como condicion.
    z: (n, latent_dim). c: (n, cond_dim). Returns: (n, latent_dim + cond_dim).
    """
    return np.concatenate([z, c], axis=axis)


def conditional_generator(z, c, W1, b1, W2, b2):
    """G(z, c): concat z y c, MLP."""
    h = concat_condition(z, c)
    h = np.maximum(0, h @ W1 + b1)
    return _sigmoid(h @ W2 + b2)


def conditional_discriminator(x, c, W1, b1, W2, b2):
    """D(x, c): concat x y c, MLP -> logit."""
    h = concat_condition(x, c)
    h = np.maximum(0, h @ W1 + b1)
    return h @ W2 + b2


def projection_discriminator(x, c, W_proj, W1, b1, W2, b2):
    """Projection D (cGAN, Miyato 2018): D(x) + c @ W_proj.
    Mas estable que concat para clases.
    """
    d = np.maximum(0, x @ W1 + b1) @ W2 + b2
    proj = c @ W_proj
    return d + proj


def patch_gan_output(h, w):
    """PatchGAN: output (h, w) con sigmoid, evalua por patch."""
    return np.zeros((h, w))  # placeholder, en realidad conv produce esto


def pix2pix_l1_loss(fake, real):
    """pix2pix usa L1 (no L2) entre fake y real para sharpness."""
    return float(np.abs(fake - real).mean())


def cgan_g_loss(d_fake_logits):
    """CGAN generator: fool D, conditional."""
    return float(-d_fake_logits.mean())


def cgan_d_loss(d_real_logits, d_fake_logits):
    """CGAN discriminator: real -> 1, fake -> 0."""
    eps = 1e-9
    d_real_loss = -np.log(_sigmoid(d_real_logits) + eps).mean()
    d_fake_loss = -np.log(1 - _sigmoid(d_fake_logits) + eps).mean()
    return float(d_real_loss + d_fake_loss)


def main() -> int:
    n, latent, c_dim, out_dim = 4, 8, 4, 16
    z = np.random.default_rng(0).standard_normal((n, latent))
    c = np.random.default_rng(1).standard_normal((n, c_dim))
    W_g1 = np.random.default_rng(2).standard_normal((latent + c_dim, 16)) * 0.1
    b_g1 = np.zeros(16)
    W_g2 = np.random.default_rng(3).standard_normal((16, out_dim)) * 0.1
    b_g2 = np.zeros(out_dim)
    fake = conditional_generator(z, c, W_g1, b_g1, W_g2, b_g2)
    print(f"CGAN fake: {fake.shape}")
    # L1 loss
    real = np.random.default_rng(4).standard_normal((n, out_dim))
    l1 = pix2pix_l1_loss(fake, real)
    print(f"pix2pix L1 loss: {l1:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())