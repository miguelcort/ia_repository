"""
Lección: 07-texto-a-habla-tts
Fase: 06
Prerrequisitos: 05-arquitectura-whisper-y-fine-tuning
"""
from __future__ import annotations
import sys
import numpy as np


def text_to_phonemes_mock(texto):
    """Mock: texto a fonemas (phoneme conversion)."""
    return [c for c in texto.lower() if c.isalpha()]


def mel_spectrogram_tts(texto, n_mels=80, n_frames=200):
    """Mock: mel-spectrogram target (lo que el TTS predice)."""
    return np.random.default_rng(0).normal(size=(n_mels, n_frames))


def vocoder_mock(mel_spec):
    """Mock: vocoder convierte mel-spec a waveform.
    En produccion: HiFi-GAN, WaveNet, UnivNet."""
    n_samples = mel_spec.shape[1] * 256
    return np.random.default_rng(0).normal(size=(n_samples,)).astype(np.float32)


def pitch_extraction_mock(senal, sample_rate=22050):
    """Mock: extraccion de pitch (F0) basica via autocorrelation."""
    if len(senal) < 100:
        return 0
    autocorr = np.correlate(senal[:1000], senal[:1000], mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    if autocorr[0] == 0:
        return 0
    autocorr /= autocorr[0]
    # Encuentra primer pico despues de cierto lag
    for lag in range(50, 500):
        if autocorr[lag] > 0.5 and (lag == 0 or autocorr[lag] > autocorr[lag - 1]):
            return sample_rate / lag
    return 0


def main() -> int:
    texto = "Hola mundo"
    phonemes = text_to_phonemes_mock(texto)
    print(f"Fonemas ({len(phonemes)}): {phonemes[:10]}")
    mel = mel_spectrogram_tts(texto)
    print(f"Mel-spec: {mel.shape}")
    audio = vocoder_mock(mel)
    print(f"Audio: {audio.shape}")
    pitch = pitch_extraction_mock(audio)
    print(f"Pitch: {pitch:.1f} Hz")
    return 0


if __name__ == "__main__":
    sys.exit(main())