"""
Lección: 20-transformada-fourier
Fase: 01
Prerrequisitos: 19-numeros-complejos
"""
from __future__ import annotations
import sys
import numpy as np


def fft(senal):
    return np.fft.fft(senal)


def ifft(espectro):
    return np.fft.ifft(espectro)


def fft_frecuencias(N, fs=1.0):
    return np.fft.fftfreq(N, d=1 / fs)


def fourier_demostrar(senal, fs=1.0):
    """Devuelve frecuencias, magnitud y fase del espectro."""
    espectro = fft(senal)
    freqs = fft_frecuencias(len(senal), fs)
    magnitud = np.abs(espectro)
    fase = np.angle(espectro)
    return freqs, magnitud, fase


def conv_fft(x, y):
    """Convolucion circular via FFT (para senales pequenas)."""
    n = len(x) + len(y) - 1
    N = 1
    while N < n:
        N *= 2
    X = np.fft.fft(x, n=N)
    Y = np.fft.fft(y, n=N)
    return np.fft.ifft(X * Y)[:n]


def main() -> int:
    fs = 100.0
    t = np.linspace(0, 1, int(fs), endpoint=False)
    senal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 12 * t)
    freqs, mag, fase = fourier_demostrar(senal, fs)
    idx_5 = np.argmin(np.abs(freqs - 5))
    idx_12 = np.argmin(np.abs(freqs - 12))
    print(f"Magnitud en f=5 Hz: {mag[idx_5]:.2f} (esperado ~50)")
    print(f"Magnitud en f=12 Hz: {mag[idx_12]:.2f} (esperado ~25)")
    return 0


if __name__ == "__main__":
    sys.exit(main())