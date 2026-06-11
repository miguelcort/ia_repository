"""
Lección: 10-modelos-de-lenguaje-de-audio
Fase: 06
Prerrequisitos: 09-generacion-de-musica
"""
from __future__ import annotations
import sys
import numpy as np


def encode_audio_tokens(audio, n_codebooks=4, codebook_size=512):
    """Mock: codifica audio a tokens (EnCodec, SoundStream, EnCodec 24kHz).
    Cuantizacion VQ multi-codebook."""
    n_samples = len(audio)
    hop = 320  # EnCodec hop: 12.5ms
    n_frames = n_samples // hop
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, codebook_size, size=(n_codebooks, n_frames))
    return tokens


def decode_audio_tokens(tokens, hop=320, sample_rate=24000):
    """Mock: decodifica tokens -> waveform."""
    n_frames = tokens.shape[1]
    return np.random.default_rng(1).normal(size=(n_frames * hop,)).astype(np.float32)


def multimodal_embedding_audio(audio, dim=1024):
    """Mock: audio embedding (CLAP, ImageBind)."""
    rng = np.random.default_rng(0)
    return rng.normal(0, 1, size=(dim,))


def text_embedding(text, dim=1024):
    """Mock: text embedding (CLIP, T5)."""
    rng = np.random.default_rng(hash(text) % 2**32)
    return rng.normal(0, 1, size=(dim,))


def clap_similarity(audio_emb, text_emb):
    """Similitud coseno entre embeddings audio y texto."""
    na = np.linalg.norm(audio_emb)
    nt = np.linalg.norm(text_emb)
    if na == 0 or nt == 0:
        return 0.0
    return float(audio_emb @ text_emb / (na * nt))


def main() -> int:
    audio = np.random.default_rng(0).normal(size=24000)  # 1s
    tokens = encode_audio_tokens(audio)
    print(f"Audio tokens: {tokens.shape}")
    decoded = decode_audio_tokens(tokens)
    print(f"Decoded: {decoded.shape}")
    audio_emb = multimodal_embedding_audio(audio)
    text_emb = text_embedding("un perro ladrando")
    sim = clap_similarity(audio_emb, text_emb)
    print(f"CLAP similarity: {sim:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())