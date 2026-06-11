"""
Lección: 13-codecs-de-audio-neurales
Fase: 06
Prerrequisitos: 12-pipeline-de-asistente-de-voz
"""
from __future__ import annotations
import sys
import numpy as np


def encodec_compress_mock(audio, n_codebooks=4, codebook_size=512, sample_rate=24000, hop=320):
    """Mock: neural codec (EnCodec, SoundStream).
    Comprime audio a tokens discretos (~ 24kHz -> 75 Hz tokens, 320x compression)."""
    n_frames = len(audio) // hop
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, codebook_size, size=(n_codebooks, n_frames))
    return tokens, {"bitrate_kbps": 6.0}


def decodec_decompress_mock(tokens, hop=320, sample_rate=24000):
    """Mock: decoder de neural codec. Devuelve waveform."""
    n_frames = tokens.shape[1]
    return np.random.default_rng(1).normal(size=(n_frames * hop,)).astype(np.float32)


def bitrate_kbps(n_codebooks, codebook_size, hop, sample_rate):
    """Bitrate en kbps: codebook_bits * n_codebooks / hop * sample_rate / 1000."""
    codebook_bits = int(np.log2(codebook_size))
    return (codebook_bits * n_codebooks * sample_rate / hop) / 1000


def compress_ratio(sample_rate, hop):
    """Ratio de compresion: cuanto menor, mas compression."""
    return sample_rate / hop


def main() -> int:
    sr = 24000
    audio = np.random.default_rng(0).normal(size=sr)  # 1s
    tokens, info = encodec_compress_mock(audio, n_codebooks=4, codebook_size=512)
    print(f"Audio: {len(audio)} samples, Tokens: {tokens.shape}, Bitrate: {info['bitrate_kbps']} kbps")
    print(f"Compress ratio: {compress_ratio(sr, 320)}x")
    print(f"Bitrate formula: {bitrate_kbps(4, 512, 320, sr)} kbps")
    return 0


if __name__ == "__main__":
    sys.exit(main())