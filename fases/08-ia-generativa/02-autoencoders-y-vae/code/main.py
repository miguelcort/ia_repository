"""
Lección: 02-autoencoders-y-vae
Fase: 08
Autoencoder: encoder -> bottleneck -> decoder. VAE: encoder mu, logvar + reparam.
"""
from __future__ import annotations
import sys
import numpy as np


def reparameterize(mu, logvar, seed=0):
    """Reparameterization trick: z = mu + sigma * eps, eps ~ N(0,1)."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(mu.shape)
    return mu + np.exp(0.5 * logvar) * eps


def kl_divergence(mu, logvar):
    """KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))."""
    return -0.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar))


def reconstruction_loss(x, x_recon, reduction="mean"):
    """MSE entre input y reconstruction."""
    err = (x - x_recon) ** 2
    if reduction == "mean":
        return err.mean()
    elif reduction == "sum":
        return err.sum()
    return err


def vae_loss(x, x_recon, mu, logvar):
    """ELBO loss: recon + KL."""
    recon = reconstruction_loss(x, x_recon)
    kl = kl_divergence(mu, logvar)
    return recon + kl, recon, kl


def encode(x, W_enc, b_enc):
    return np.tanh(x @ W_enc + b_enc)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode(z, W_dec, b_dec):
    return _sigmoid(z @ W_dec + b_dec)


def vae_forward(x, W_enc, b_enc, W_mu, b_mu, W_logvar, b_logvar,
                W_dec, b_dec, seed=0):
    """Forward: x -> h -> mu, logvar -> z -> x_recon."""
    h = encode(x, W_enc, b_enc)
    mu = h @ W_mu + b_mu
    logvar = h @ W_logvar + b_logvar
    z = reparameterize(mu, logvar, seed=seed)
    x_recon = decode(z, W_dec, b_dec)
    return x_recon, mu, logvar, z


def sample_prior(latent_dim, n_samples, seed=0):
    """Sample z ~ N(0, 1)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, latent_dim))


def main() -> int:
    latent_dim = 4
    d = 8
    rng = np.random.default_rng(0)
    W_enc = rng.standard_normal((d, 16)) * 0.1
    b_enc = np.zeros(16)
    W_mu = rng.standard_normal((16, latent_dim)) * 0.1
    b_mu = np.zeros(latent_dim)
    W_logvar = rng.standard_normal((16, latent_dim)) * 0.1
    b_logvar = np.zeros(latent_dim)
    W_dec = rng.standard_normal((latent_dim, d)) * 0.1
    b_dec = np.zeros(d)
    x = rng.standard_normal((3, d)) * 0.5
    x_recon, mu, logvar, z = vae_forward(
        x, W_enc, b_enc, W_mu, b_mu, W_logvar, b_logvar, W_dec, b_dec)
    print(f"Input: {x.shape}, Recon: {x_recon.shape}")
    print(f"Latent mu: shape {mu.shape}, range [{mu.min():.3f}, {mu.max():.3f}]")
    print(f"KL div: {kl_divergence(mu, logvar):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())