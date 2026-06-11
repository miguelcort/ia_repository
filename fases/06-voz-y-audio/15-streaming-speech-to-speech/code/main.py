"""
Lección: 15-streaming-speech-to-speech
Fase: 06
Prerrequisitos: 14-deteccion-de-actividad-de-voz-y-turn-taking
"""
from __future__ import annotations
import sys
import numpy as np


def streaming_asr_to_llm_pipeline(audio_chunk, sample_rate=16000):
    """Mock: streaming speech -> LLM pipeline.
    Devuelve respuesta parcial."""
    if len(audio_chunk) < sample_rate // 4:  # min 250ms
        return None
    texto = " ".join(["palabra"] * max(1, len(audio_chunk) // (sample_rate // 2)))
    return texto


def tts_streaming_tokens(text, codec="encodec"):
    """Mock: text -> streaming codec tokens para TTS.
    Genera tokens que pueden ser 'streameados' al decoder."""
    n_chars = len(text)
    # 75 tokens/sec * 1 sec per 4 chars (mock) = ~18 tokens
    n_tokens = max(1, n_chars // 4 * 75 // 10)
    return np.random.default_rng(0).integers(0, 1024, size=(n_tokens,)).tolist()


def latency_full_duplex(audio_chunk_ms=100, asr_latency_ms=200, llm_ttft_ms=300, tts_first_chunk_ms=200):
    """Latencia total = audio chunk / 2 + ASR + LLM TTFT + TTS first chunk.
    Para experiencia natural: < 800ms."""
    return audio_chunk_ms / 2 + asr_latency_ms + llm_ttft_ms + tts_first_chunk_ms


def main() -> int:
    audio = np.random.default_rng(0).normal(size=16000)  # 1s
    # Streaming ASR -> LLM
    texto = streaming_asr_to_llm_pipeline(audio)
    print(f"ASR streaming: {texto[:30] if texto else 'None'}")
    # TTS streaming
    tokens = tts_streaming_tokens("Hola mundo")
    print(f"TTS tokens: {len(tokens)}")
    # Latency
    lat = latency_full_duplex()
    print(f"Latencia full-duplex: {lat:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())