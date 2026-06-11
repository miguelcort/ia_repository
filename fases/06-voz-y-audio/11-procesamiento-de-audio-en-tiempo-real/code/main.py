"""
Lección: 11-procesamiento-de-audio-en-tiempo-real
Fase: 06
Prerrequisitos: 10-modelos-de-lenguaje-de-audio
"""
from __future__ import annotations
import sys
import time
import numpy as np


def vad_silero_mock(audio, sample_rate=16000, threshold=0.5):
    """Mock: Voice Activity Detection. Devuelve 1 si habla, 0 silencio."""
    if len(audio) == 0:
        return 0
    rms = float(np.sqrt(np.mean(audio ** 2)))
    # Heuristica: RMS > threshold indica habla
    return 1 if rms > 0.01 else 0


def streaming_asr_chunk(audio, sample_rate=16000, chunk_seconds=0.5):
    """Mock: ASR streaming. Procesa audio en chunks de 0.5s,
    devuelve parciales (latency vs accuracy)."""
    chunk_samples = int(chunk_seconds * sample_rate)
    resultados = []
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        # Mock: transcripcion proporcional al tamano
        n_words = max(1, len(chunk) // (sample_rate // 2))  # ~2 words/sec
        resultados.append(" ".join(["palabra"] * n_words))
    return resultados


def latency_streaming(model_latency_ms=300, chunk_ms=500):
    """Calcula la latencia de un sistema streaming."""
    return model_latency_ms + chunk_ms / 2


def main() -> int:
    sr = 16000
    audio_habla = np.random.default_rng(0).normal(scale=0.1, size=sr)  # habla
    audio_silencio = np.zeros(sr)  # silencio
    print(f"VAD habla: {vad_silero_mock(audio_habla)}")
    print(f"VAD silencio: {vad_silero_mock(audio_silencio)}")
    # Streaming ASR
    chunks = streaming_asr_chunk(audio_habla, sr, chunk_seconds=0.5)
    print(f"Chunks: {len(chunks)}")
    print(f"Latencia: {latency_streaming()} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())