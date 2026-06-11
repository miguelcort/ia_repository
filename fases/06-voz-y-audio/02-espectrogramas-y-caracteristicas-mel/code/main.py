"""
Lección: 02-espectrogramas-y-caracteristicas-mel
Fase: 06
Prerrequisitos: 01-fundamentos-de-audio
"""
from __future__ import annotations
import sys
import numpy as np


def fft_simple(senal):
    """FFT basica (mock de numpy.fft)."""
    N = len(senal)
    # Implementacion naive: O(N^2). Solo para test.
    out = np.zeros(N, dtype=complex)
    for k in range(N):
        s = 0
        for n in range(N):
            s += senal[n] * np.exp(-2j * np.pi * k * n / N)
        out[k] = s
    return out


def stft_simple(senal, ventana=400, hop=160, n_fft=512):
    """Short-Time Fourier Transform. Devuelve |STFT|^2 (potencia)."""
    n_frames = (len(senal) - ventana) // hop + 1
    out = np.zeros((n_fft // 2 + 1, n_frames))
    win = np.hanning(ventana)
    for i in range(n_frames):
        frame = senal[i * hop:i * hop + ventana] * win
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        spec = np.abs(fft_simple(frame)[:n_fft // 2 + 1]) ** 2
        out[:, i] = spec
    return out


def hz_a_mel(freq):
    """Conversion Hz a escala Mel (formula popular: O'Shaughnessy)."""
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def mel_a_hz(mel):
    """Conversion Mel a Hz."""
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def mel_filterbank(n_mels=80, n_fft=512, sample_rate=16000):
    """Crea el banco de filtros triangulares en escala Mel.
    Devuelve (n_mels, n_fft//2+1)."""
    f_min = 0
    f_max = sample_rate / 2
    mel_min = hz_a_mel(f_min)
    mel_max = hz_a_mel(f_max)
    # n_mels + 2 puntos equiespaciados en mel
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([mel_a_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    # Construir filtros triangulares
    filters = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        left, center, right = bin_points[m], bin_points[m + 1], bin_points[m + 2]
        for k in range(left, center):
            if center > left:
                filters[m, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right > center:
                filters[m, k] = (right - k) / (right - center)
    return filters


def mel_spectrogram(senal, sample_rate=16000, n_mels=80, n_fft=400, hop=160):
    """Pipeline: STFT -> Mel filterbank -> log."""
    spec = stft_simple(senal, ventana=n_fft, hop=hop, n_fft=n_fft)
    filters = mel_filterbank(n_mels=n_mels, n_fft=n_fft, sample_rate=sample_rate)
    mel_spec = filters @ spec
    # log-mel
    return np.log(mel_spec + 1e-9)


def main() -> int:
    # Senal de prueba: 440Hz + 880Hz
    sr = 16000
    t = np.arange(0, 1.0, 1 / sr)
    senal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
    mel = mel_spectrogram(senal, sample_rate=sr)
    print(f"Mel-spec shape: {mel.shape}")  # (80, ~100)
    return 0


if __name__ == "__main__":
    sys.exit(main())