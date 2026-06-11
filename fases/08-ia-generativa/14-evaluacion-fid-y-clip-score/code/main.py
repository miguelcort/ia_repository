"""
Lección: 14-evaluacion-fid-y-clip-score
Fase: 08
FID: Frechet Inception Distance. CLIP score: text-image alignment.
"""
from __future__ import annotations
import sys
import numpy as np


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Frechet distance entre dos Gaussianas.
    FD = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * sqrt(sigma1 @ sigma2))
    Implementacion sin scipy.linalg.sqrtm: usa eigendecomp.
    """
    diff = mu1 - mu2
    # sqrt(sigma1 @ sigma2) via eigendecomp de la matriz producto
    prod = sigma1 @ sigma2
    # Asegurar simetria
    prod = (prod + prod.T) / 2
    eigvals, eigvecs = np.linalg.eigh(prod)
    # Clip para no negatividad
    eigvals = np.maximum(eigvals, 0)
    covmean = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    fd = diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return float(fd)


def fid_score(real_features, fake_features):
    """FID: features son activaciones Inception pool3.
    Calcula mu, sigma por set y Frechet distance.
    """
    mu_real = real_features.mean(axis=0)
    mu_fake = fake_features.mean(axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)
    return frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)


def inception_score(logits, eps=1e-9):
    """IS: exp(E[KL(p(y|x) || p(y))]).
    logits: (n_samples, n_classes). Requiere softmax.
    """
    # p(y|x)
    p_yx = np.exp(logits - logits.max(axis=-1, keepdims=True))
    p_yx = p_yx / p_yx.sum(axis=-1, keepdims=True)
    # p(y) = E_x[p(y|x)]
    p_y = p_yx.mean(axis=0)
    # KL
    kl = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
    return float(np.exp(kl.sum(axis=-1).mean()))


def clip_score(image_features, text_features):
    """CLIP score: cosine similarity entre image y text features.
    """
    image_features = image_features / (np.linalg.norm(image_features, axis=-1, keepdims=True) + 1e-9)
    text_features = text_features / (np.linalg.norm(text_features, axis=-1, keepdims=True) + 1e-9)
    return float((image_features * text_features).sum(axis=-1).mean())


def precision_recall(real_features, fake_features, k=3):
    """Precision y recall via k-NN manifolds (Kynkaanniemi 2019).
    precision: fraccion de fake dentro del manifold de real.
    recall: fraccion de real dentro del manifold de fake.
    """
    def manifold_distance(features, k):
        # Para cada punto, distancia al k-NN
        n = features.shape[0]
        dists = np.zeros(n)
        for i in range(n):
            d = np.linalg.norm(features - features[i], axis=-1)
            d.sort()
            dists[i] = d[k]
        return dists

    real_radii = manifold_distance(real_features, k)
    fake_radii = manifold_distance(fake_features, k)
    # Precision: para cada fake, esta dentro de real manifold?
    precision = 0
    for f in fake_features:
        d = np.linalg.norm(real_features - f, axis=-1)
        if (d < real_radii).any():
            precision += 1
    precision /= len(fake_features)
    # Recall: para cada real, esta dentro de fake manifold?
    recall = 0
    for r in real_features:
        d = np.linalg.norm(fake_features - r, axis=-1)
        if (d < fake_radii).any():
            recall += 1
    recall /= len(real_features)
    return precision, recall


def metric_summary():
    """Resumen de metricas."""
    return {
        "FID": "Frechet Inception Distance. Real vs fake. Mas bajo = mejor.",
        "IS": "Inception Score. Calidad y diversity. Mas alto = mejor.",
        "CLIP score": "Text-image alignment. 0-1 (cosine sim).",
        "Precision/Recall": "Mode coverage. Precision = calidad, Recall = coverage.",
        "LPIPS": "Perceptual distance. 0-1.",
        "Human eval": "Gold standard, costoso.",
    }


def main() -> int:
    print("=== Metric summary ===")
    for k, v in metric_summary().items():
        print(f"  {k:18s} {v}")
    # Demo
    n_real, n_fake, d = 100, 100, 16
    rng = np.random.default_rng(0)
    real = rng.standard_normal((n_real, d))
    fake = rng.standard_normal((n_fake, d)) * 1.1 + 0.1
    fid = fid_score(real, fake)
    print(f"\nFID: {fid:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())