"""
Lección: 17-metricas-de-evaluacion-de-audio
Fase: 06
Prerrequisitos: 16-anti-spoofing-y-audio-watermarking
"""
from __future__ import annotations
import sys
import numpy as np


def pesq_score(ref, deg, sample_rate=16000):
    """PESQ (Perceptual Eval of Speech Quality) mock: range -0.5 to 4.5.
    4.5 = perfecto, 1.0 = malo. ITU-T P.862."""
    if len(ref) != len(deg):
        # Truncar al mas corto
        n = min(len(ref), len(deg))
        ref = ref[:n]
        deg = deg[:n]
    # Mock: correlacion con ruido
    if np.std(ref) == 0 or np.std(deg) == 0:
        return 1.0
    corr = float(np.corrcoef(ref, deg)[0, 1])
    return max(1.0, min(4.5, 2.0 + 2 * corr))


def stoi_score(ref, deg, sample_rate=16000):
    """STOI (Short-Time Objective Intelligibility) mock: 0 a 1.
    1 = perfectamente inteligible, 0 = no se entiende."""
    if len(ref) != len(deg):
        n = min(len(ref), len(deg))
        ref = ref[:n]
        deg = deg[:n]
    # Mock: correlacion con ruido
    if np.std(ref) == 0 or np.std(deg) == 0:
        return 0.0
    return float(max(0.0, min(1.0, abs(np.corrcoef(ref, deg)[0, 1]))))


def visqol_score(ref, deg, sample_rate=16000):
    """ViSQOL (Virtual Speech Quality Objective Listener) mock: 1 a 5.
    MOS-like."""
    pesq = pesq_score(ref, deg, sample_rate)
    return 1.0 + pesq / 4.5 * 4  # mapear a 1-5


def mos_estimate(ref, deg, sample_rate=16000):
    """Estimacion de MOS (Mean Opinion Score) mock: 1-5."""
    pesq = pesq_score(ref, deg, sample_rate)
    # PESQ 4.5 = MOS 5; PESQ 1.0 = MOS 1
    return 1.0 + (pesq - 1.0) / 3.5 * 4


def fad_score(embeddings_real, embeddings_fake):
    """FAD (Frechet Audio Distance) mock.
    Menor = mas cerca de real."""
    mu_r = np.mean(embeddings_real, axis=0)
    mu_f = np.mean(embeddings_fake, axis=0)
    diff = mu_r - mu_f
    return float(np.dot(diff, diff))


def main() -> int:
    sr = 16000
    ref = np.random.default_rng(0).normal(scale=0.1, size=sr)
    deg = ref + 0.01 * np.random.default_rng(1).normal(size=sr)  # casi igual
    print(f"PESQ: {pesq_score(ref, deg):.3f}")
    print(f"STOI: {stoi_score(ref, deg):.3f}")
    print(f"ViSQOL: {visqol_score(ref, deg):.3f}")
    print(f"MOS: {mos_estimate(ref, deg):.3f}")
    # FAD
    emb_real = np.random.default_rng(0).normal(size=(10, 64))
    emb_fake = np.random.default_rng(1).normal(size=(10, 64))
    print(f"FAD: {fad_score(emb_real, emb_fake):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())