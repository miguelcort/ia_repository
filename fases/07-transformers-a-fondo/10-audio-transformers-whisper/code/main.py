"""
Lección: 10-audio-transformers-whisper
Fase: 07
Whisper: encoder-decoder para ASR. Audio -> log-mel -> encoder -> decoder -> text.
"""
from __future__ import annotations
import sys
import numpy as np


def log_mel_spectrogram(audio, n_mels=80, n_fft=400, hop=160, sample_rate=16000):
    """Audio -> log-mel spectrogram (mock: averaging en ventanas)."""
    # Padding a multiple of hop
    n = len(audio)
    pad = (hop - n % hop) % hop
    audio = np.concatenate([audio, np.zeros(pad)])
    # Frames
    n_frames = len(audio) // hop
    frames = audio[:n_frames * hop].reshape(n_frames, hop)
    # Energy por frame (proxy para mel)
    energy = np.abs(frames).mean(axis=-1)  # (n_frames,)
    # Mel: aplicar peso aleatorio a mel bands (mock)
    rng = np.random.default_rng(0)
    mel_basis = np.abs(rng.standard_normal((n_mels, n_frames))) * 0.1
    mel = mel_basis * energy.reshape(1, -1)
    return np.log(mel + 1e-6).T  # (n_frames, n_mels)


def conv_embedding(mel, W, b):
    """Conv1d stride-2 embedding: 2x downsample in time."""
    # (T, n_mels) -> matmul con kernel stride 2
    T, M = mel.shape
    if T % 2 == 1:
        mel = np.vstack([mel, np.zeros((1, M))])
        T += 1
    pooled = (mel[0::2] + mel[1::2]) / 2  # avg pool
    return pooled @ W.T + b


def sinusoidal_pe(seq_len, d_model):
    pos = np.arange(seq_len).reshape(-1, 1)
    i = np.arange(d_model).reshape(1, -1)
    angle = pos / (10000 ** (2 * (i // 2) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe


def whisper_transcribe_mock(audio, n_mels=80):
    """Mock pipeline: audio -> log-mel -> encoder -> decoder -> text ids."""
    mel = log_mel_spectrogram(audio, n_mels=n_mels)
    # Encoder: reduce 2x con conv
    d_model = 64
    W = np.random.default_rng(0).standard_normal((d_model, n_mels)) * 0.02
    b = np.zeros(d_model)
    enc = conv_embedding(mel, W, b)
    enc = enc + sinusoidal_pe(enc.shape[0], d_model)
    # Decoder: retorna secuencia corta (mock)
    text_len = 8
    dec_out = np.random.default_rng(1).standard_normal((text_len, d_model))
    return enc, dec_out


def decode_to_text(token_ids, vocab):
    """Mock decode: token_id -> word."""
    inv_vocab = {i: w for w, i in vocab.items()}
    return " ".join(inv_vocab.get(t, "<unk>") for t in token_ids)


def main() -> int:
    sample_rate = 16000
    audio = np.random.default_rng(0).standard_normal(sample_rate * 3)  # 3s
    mel = log_mel_spectrogram(audio)
    print(f"Audio 3s: {len(audio)} samples")
    print(f"Mel: {mel.shape} (T, n_mels=80)")
    enc, dec = whisper_transcribe_mock(audio)
    print(f"Encoder: {enc.shape} (T/2, d_model=64)")
    print(f"Decoder: {dec.shape} (text_len, d_model)")
    return 0


if __name__ == "__main__":
    sys.exit(main())