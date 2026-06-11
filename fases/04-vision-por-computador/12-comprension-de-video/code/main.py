"""
Lección: 12-comprension-de-video
Fase: 04
Prerrequisitos: 11-stable-diffusion
"""
from __future__ import annotations
import sys
import numpy as np


def leer_video_mock(n_frames=10, H=8, W=8):
    """Genera secuencia de video sintetica (frames cambian ligeramente)."""
    rng = np.random.default_rng(0)
    video = []
    base = rng.normal(0, 1, size=(H, W))
    for t in range(n_frames):
        frame = base + 0.1 * t * np.random.default_rng(t).normal(0, 0.5, size=(H, W))
        video.append(frame)
    return np.array(video)


def muestrear_frames(video, n=5, estrategia="uniforme"):
    """Toma n frames del video. Estrategias: uniforme, aleatorio, primer/ultimo + medio."""
    T = len(video)
    if estrategia == "uniforme":
        idx = np.linspace(0, T - 1, n, dtype=int)
    elif estrategia == "aleatorio":
        idx = np.sort(np.random.default_rng(0).choice(T, n, replace=False))
    elif estrategia == "key":
        # Primer + ultimo + N intermedios
        idx = np.linspace(0, T - 1, n, dtype=int)
    else:
        idx = np.arange(n)
    return video[idx], idx


def optical_flow_fake(frame_a, frame_b):
    """Mock: diferencia entre frames como proxy de optical flow."""
    return frame_b - frame_a


def diferencia_entre_frames(video):
    """Calcula diferencia promedio entre frames consecutivos."""
    diffs = []
    for t in range(len(video) - 1):
        d = float(np.abs(video[t + 1] - video[t]).mean())
        diffs.append(d)
    return diffs


def main() -> int:
    video = leer_video_mock(n_frames=20)
    print(f"Video shape: {video.shape}")
    frames, idx = muestrear_frames(video, n=5)
    print(f"Sampled: {idx}, shape: {frames.shape}")
    diffs = diferencia_entre_frames(video)
    print(f"Diffs (primeros 5): {diffs[:5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())