"""
Lección: 05-arquitectura-whisper-y-fine-tuning
Fase: 06
Prerrequisitos: 04-reconocimiento-de-habla-asr
"""
from __future__ import annotations
import sys
import numpy as np


def chunk_audio(senal, max_seconds=30, sample_rate=16000):
    """Divide audio en chunks de max_seconds.
    Whisper usa 30s chunks."""
    max_samples = max_seconds * sample_rate
    n_chunks = (len(senal) + max_samples - 1) // max_samples
    chunks = []
    for i in range(n_chunks):
        chunk = senal[i * max_samples:(i + 1) * max_samples]
        if len(chunk) < max_samples:
            # Padding con zeros
            chunk = np.pad(chunk, (0, max_samples - len(chunk)))
        chunks.append(chunk)
    return chunks


def mel_spectrogram_whisper(senal, n_mels=80, n_fft=400, hop=160, sample_rate=16000):
    """Mock: mel-spectrogram estilo Whisper (80 mels, log-scale)."""
    # En produccion: librosa.feature.melspectrogram + log
    n_frames = (len(senal) - n_fft) // hop + 1
    return np.random.default_rng(0).normal(size=(n_mels, max(1, n_frames)))


def tokenizar_texto_whisper(texto):
    """Mock: tokenizacion BPE estilo Whisper. Devuelve lista de ids."""
    bpe_vocab = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz ")}
    return [bpe_vocab.get(c, 0) for c in texto.lower()]


def decoder_greedy_mock(logits):
    """Mock: greedy decoder."""
    return int(logits.argmax())


def special_tokens_whisper():
    """Tokens especiales de Whisper."""
    return {
        "<|startoftranscript|>": 50257,
        "<|endoftext|>": 50256,
        "<|transcribe|>": 50358,
        "<|translate|>": 50357,
        "<|notimestamps|>": 50363,
        "<|es|>": 50262,  # Spanish language token
    }


def main() -> int:
    # Audio de 1 minuto
    sr = 16000
    senal = np.random.default_rng(0).normal(size=sr * 60)
    chunks = chunk_audio(senal, max_seconds=30, sample_rate=sr)
    print(f"Chunks: {len(chunks)}")
    for i, c in enumerate(chunks):
        mel = mel_spectrogram_whisper(c)
        print(f"  Chunk {i}: mel shape {mel.shape}")
    print(f"\nSpecial tokens: {special_tokens_whisper()}")
    print(f"Tokens de 'hola': {tokenizar_texto_whisper('hola')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())