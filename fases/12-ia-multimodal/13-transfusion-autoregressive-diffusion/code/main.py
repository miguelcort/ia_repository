"""
Lección: 13-transfusion-autoregressive-diffusion
Fase: 12
TransFusion (Zhou 2024): predice siguiente token + diffusion en un solo modelo.
Text autoregresivo + image diffusion. Shared transformer backbone.
"""
from __future__ import annotations
import numpy as np


def text_token_ids(text, vocab_size):
    """Mock text -> ids."""
    return np.array([hash(c) % vocab_size for c in text], dtype=np.int64)


def diffusion_forward(noisy_image, t, model, sigma_schedule):
    """Predict noise in noisy image. (H, W, C) -> (H, W, C)."""
    rng = np.random.default_rng(t)
    return rng.standard_normal(noisy_image.shape) * 0.1


def diffusion_denoise_step(image, t, predicted_noise, alpha_t, sigma_t):
    """Single denoising step: x_t -> x_{t-1}."""
    x0_pred = (image - sigma_t * predicted_noise) / alpha_t
    return alpha_t * x0_pred + sigma_t * predicted_noise


def add_noise(image, t, noise_schedule):
    """Add Gaussian noise. (H, W, C) -> (H, W, C)."""
    rng = np.random.default_rng(t + 1)
    return image + noise_schedule[t] * rng.standard_normal(image.shape)


def transfusion_loss_text(logits, target_ids):
    """Cross-entropy for text tokens."""
    n = logits.shape[0]
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    log_probs = np.log(e / e.sum(axis=-1, keepdims=True))
    return -log_probs[np.arange(n), target_ids].mean()


def transfusion_loss_image(noisy_image, predicted_noise, true_noise):
    """MSE for image diffusion."""
    return float(np.mean((predicted_noise - true_noise) ** 2))


def transfusion_step(text_logits, target_ids, noisy_image, predicted_noise, true_noise):
    """Combined loss."""
    L_text = transfusion_loss_text(text_logits, target_ids)
    L_image = transfusion_loss_image(noisy_image, predicted_noise, true_noise)
    return L_text + L_image


def main() -> int:
    rng = np.random.default_rng(0)
    img = rng.standard_normal((64, 64, 3))
    noise_schedule = np.linspace(0.01, 0.99, 100)
    noisy = add_noise(img, t=50, noise_schedule=noise_schedule)
    pred = diffusion_forward(noisy, t=50, model=None, sigma_schedule=noise_schedule)
    text = np.random.default_rng(0).standard_normal((10, 100))
    targets = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    loss = transfusion_step(text, targets, noisy, pred, np.zeros_like(pred))
    print(f"TransFusion combined loss: {loss:.4f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())