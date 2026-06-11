"""
Lección: 12-pipeline-de-asistente-de-voz
Fase: 06
Prerrequisitos: 11-procesamiento-de-audio-en-tiempo-real
"""
from __future__ import annotations
import sys
import numpy as np


def capturar_audio(duracion=5, sample_rate=16000):
    """Mock: captura audio del microfono."""
    return np.random.default_rng(0).normal(scale=0.1, size=int(duracion * sample_rate))


def vad_silero_mock(audio, sample_rate=16000):
    """Mock: VAD."""
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return 1 if rms > 0.01 else 0


def asr_whisper(audio, sample_rate=16000):
    """Mock: ASR con Whisper."""
    n_words = max(1, len(audio) // (sample_rate // 2))
    return " ".join(["palabra"] * n_words)


def llm_respuesta(query, model="claude"):
    """Mock: LLM genera respuesta."""
    return f"Echo: dijiste '{query}'. Soy un asistente mock."


def tts_sintetizar(texto, model="openai-tts"):
    """Mock: TTS convierte texto a audio."""
    n_samples = len(texto) * 1600  # ~0.1s per char
    return np.random.default_rng(0).normal(size=(n_samples,)).astype(np.float32)


def pipeline_asistente_voz(duracion=5, sample_rate=16000):
    """Pipeline completo: captura -> VAD -> ASR -> LLM -> TTS -> output."""
    audio = capturar_audio(duracion, sample_rate)
    if vad_silero_mock(audio) == 0:
        return None, "Silencio, no se proceso."
    texto = asr_whisper(audio, sample_rate)
    respuesta = llm_respuesta(texto)
    audio_resp = tts_sintetizar(respuesta)
    return audio_resp, respuesta


def main() -> int:
    audio_resp, texto = pipeline_asistente_voz(duracion=2)
    print(f"Texto: {texto[:50]}")
    print(f"Audio respuesta shape: {audio_resp.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())