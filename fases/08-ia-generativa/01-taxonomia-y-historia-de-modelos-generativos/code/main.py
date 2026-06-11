"""
Lección: 01-taxonomia-y-historia-de-modelos-generativos
Fase: 08
Panorama: VAE, GAN, Diffusion, Flow, AR. Historia y trade-offs.
"""
from __future__ import annotations
import sys
import numpy as np


def elbo_loss(recon_loss, kl_div):
    """ELBO = -recon_loss + kl_div. Maximizar ELBO == minimizar -ELBO."""
    return recon_loss + kl_div


def gan_losses(d_real_logits, d_fake_logits):
    """Discriminator y generator losses tipicos.
    D_loss = -log(sigmoid(D(x_real))) - log(1 - sigmoid(D(x_fake)))
    G_loss = -log(sigmoid(D(x_fake)))
    """
    eps = 1e-9
    d_loss = (-np.log(_sigmoid(d_real_logits) + eps)
              - np.log(1 - _sigmoid(d_fake_logits) + eps)).mean()
    g_loss = -np.log(_sigmoid(d_fake_logits) + eps).mean()
    return float(d_loss), float(g_loss)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def diffusion_loss(predicted_noise, true_noise):
    """DDPM: ||eps - eps_theta||^2."""
    return float(((predicted_noise - true_noise) ** 2).mean())


def flow_matching_loss(predicted_velocity, target_velocity):
    """Flow matching: ||v - v_theta(x_t, t)||^2."""
    return float(((predicted_velocity - target_velocity) ** 2).mean())


def taxonomy_summary():
    """Resumen comparativo de familias generativas."""
    return [
        ("VAE", "Encoder-decoder probabilistico, ELBO, blurry"),
        ("GAN", "Adversarial, sharp, mode collapse"),
        ("AR", "Autoregresivo, transformer, likelihood exacto"),
        ("Normalizing flow", "Biyective, likelihood exacto, denso"),
        ("Diffusion", "Denoising, SOTA imagen, lento"),
        ("Flow matching", "ODE path, 10-50 steps, rapido"),
    ]


def main() -> int:
    print("=== Familias generativas ===")
    for name, desc in taxonomy_summary():
        print(f"  {name:20s} {desc}")
    # Demostrar losses
    print("\n=== Demostracion de losses ===")
    rng = np.random.default_rng(0)
    recon = 0.5
    kl = 0.3
    print(f"ELBO total: {elbo_loss(recon, kl):.3f}")
    d_loss, g_loss = gan_losses(d_real_logits=2.0, d_fake_logits=-1.0)
    print(f"GAN D_loss={d_loss:.3f}, G_loss={g_loss:.3f}")
    noise_pred = rng.standard_normal((4, 4))
    noise_true = rng.standard_normal((4, 4))
    print(f"Diffusion loss: {diffusion_loss(noise_pred, noise_true):.3f}")
    v_pred = rng.standard_normal((4, 4))
    v_true = rng.standard_normal((4, 4))
    print(f"Flow matching loss: {flow_matching_loss(v_pred, v_true):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())