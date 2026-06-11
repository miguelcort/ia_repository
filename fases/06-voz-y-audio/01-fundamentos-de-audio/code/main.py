"""
Lección: 01-fundamentos-de-audio
Fase: 06
Prerrequisitos: 02-fundamentos-ml/01-pandas-y-numpy
"""
from __future__ import annotations
import sys
import numpy as np


def generar_seno(frecuencia, duracion, sample_rate=16000, amplitud=1.0):
    """Genera una onda sinusoidal. tiempo: array 1D, devuelve array 1D."""
    t = np.arange(0, duracion, 1 / sample_rate)
    return amplitud * np.sin(2 * np.pi * frecuencia * t)


def sample_rate_valido(sr):
    """Valida sample rate (tipicos: 8000, 16000, 22050, 44100, 48000)."""
    return sr in (8000, 16000, 22050, 44100, 48000)


def duracion_samples(samples, sample_rate):
    """Duracion en segundos dado numero de samples y sample rate."""
    return samples / sample_rate


def energia_rms(senal):
    """Root Mean Square energy."""
    return float(np.sqrt(np.mean(senal ** 2)))


def zero_crossing_rate(senal):
    """Tasa de cruces por cero (proxy de contenido de frecuencia)."""
    cruces = np.abs(np.diff(np.sign(senal))).sum()
    return float(cruces / (2 * len(senal)))


def main() -> int:
    sr = 16000
    senal = generar_seno(440, 0.5, sample_rate=sr)  # 440Hz por 0.5s
    print(f"Senal shape: {senal.shape}, duracion: {duracion_samples(len(senal), sr):.2f}s")
    print(f"Energia RMS: {energia_rms(senal):.3f}")
    print(f"ZCR: {zero_crossing_rate(senal):.3f}")
    print(f"Sample rate valido: {sample_rate_valido(sr)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())