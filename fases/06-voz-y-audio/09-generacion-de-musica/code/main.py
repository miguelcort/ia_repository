"""
Lección: 09-generacion-de-musica
Fase: 06
Prerrequisitos: 07-texto-a-habla-tts
"""
from __future__ import annotations
import sys
import numpy as np


def piano_roll_mock(notas, duracion=4.0, sample_rate=44100):
    """Mock: piano roll (matriz notas x tiempo) -> audio.
    notas: lista de (pitch, start_time, duration)."""
    n_frames = int(duracion * sample_rate)
    audio = np.zeros(n_frames)
    for pitch, start, dur in notas:
        start_sample = int(start * sample_rate)
        n_samples = int(dur * sample_rate)
        if start_sample + n_samples > n_frames:
            n_samples = n_frames - start_sample
        t = np.arange(n_samples) / sample_rate
        freq = 440.0 * 2 ** ((pitch - 69) / 12)
        audio[start_sample:start_sample + n_samples] = np.sin(2 * np.pi * freq * t)
    return audio


def mel_spectrogram_audio(audio, sample_rate=44100, n_mels=128, n_fft=2048, hop=512):
    """Mock: mel-spectrogram de un audio."""
    n_frames = (len(audio) - n_fft) // hop + 1
    return np.random.default_rng(0).normal(size=(n_mels, max(1, n_frames)))


def musicgen_mock(prompt, duracion=10, sample_rate=32000):
    """Mock: text-to-music (MusicGen / AudioLDM)."""
    n_samples = duracion * sample_rate
    rng = np.random.default_rng(hash(prompt) % 2**32)
    return rng.normal(0, 0.1, size=(n_samples,)).astype(np.float32)


def main() -> int:
    # Piano roll simple: 3 notas
    notas = [(60, 0.0, 1.0), (64, 1.0, 1.0), (67, 2.0, 1.0)]  # C, E, G
    audio = piano_roll_mock(notas, duracion=4.0)
    print(f"Audio: {audio.shape}")
    mel = mel_spectrogram_audio(audio)
    print(f"Mel-spec: {mel.shape}")
    music = musicgen_mock("piano clasico, 100 bpm")
    print(f"MusicGen: {music.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())