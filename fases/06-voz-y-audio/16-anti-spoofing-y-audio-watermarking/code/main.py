"""
Lección: 16-anti-spoofing-y-audio-watermarking
Fase: 06
Prerrequisitos: 15-streaming-speech-to-speech
"""
from __future__ import annotations
import sys
import numpy as np


def detect_deepfake_mock(audio, sample_rate=16000, threshold=0.5):
    """Mock: detector de audio sintetico (RawNet, AASIST).
    Devuelve probabilidad de fake."""
    rng = np.random.default_rng(0)
    # Mock: probabilidad aleatoria
    p_fake = float(rng.random())
    return p_fake, 1 if p_fake > threshold else 0


def detect_spoofing_real_time(frames, threshold=0.5):
    """Mock: anti-spoofing en tiempo real. Procesa frames de 1s.
    Devuelve decision binaria."""
    p_fake = 0.0
    for i, frame in enumerate(frames):
        # Mock: incrementa prob fake si RMS es muy bajo (compresion)
        rms = float(np.sqrt(np.mean(frame ** 2)))
        p_fake += 0.1 if rms < 0.01 else 0
    p_fake /= max(1, len(frames))
    return 1 if p_fake > threshold else 0, p_fake


def embed_watermark(audio, message="CREATED BY AI"):
    """Mock: inserta watermark imperceptible en audio.
    AudioSeal (Meta 2024), SilentCipher, etc."""
    rng = np.random.default_rng(hash(message) % 2**32)
    # Mock: pseudo-random noise basado en el mensaje
    watermark = 0.001 * rng.choice([-1, 1], size=len(audio))
    return audio + watermark


def detect_watermark(audio_watermarked, message="CREATED BY AI"):
    """Mock: detecta si el audio tiene watermark.
    Calcula correlacion entre el audio y el watermark pattern."""
    rng = np.random.default_rng(hash(message) % 2**32)
    pattern = 0.001 * rng.choice([-1, 1], size=len(audio_watermarked))
    # Correlacion simple
    if np.std(audio_watermarked) > 0 and np.std(pattern) > 0:
        corr = float(np.corrcoef(audio_watermarked, pattern)[0, 1])
    else:
        corr = 0.0
    return corr > 0.1  # threshold


def main() -> int:
    audio_real = np.random.default_rng(0).normal(scale=0.1, size=16000)
    audio_fake = np.random.default_rng(1).normal(scale=0.05, size=16000)  # diferente
    p_fake_real, _ = detect_deepfake_mock(audio_real)
    p_fake_fake, _ = detect_deepfake_mock(audio_fake)
    print(f"P(fake|real): {p_fake_real:.3f}")
    print(f"P(fake|fake): {p_fake_fake:.3f}")
    # Watermarking
    audio_wm = embed_watermark(audio_fake)
    detected = detect_watermark(audio_wm)
    print(f"Watermark detectado: {detected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())