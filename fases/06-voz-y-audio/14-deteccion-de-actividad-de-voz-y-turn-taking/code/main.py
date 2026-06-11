"""
Lección: 14-deteccion-de-actividad-de-voz-y-turn-taking
Fase: 06
Prerrequisitos: 13-codecs-de-audio-neurales
"""
from __future__ import annotations
import sys
import numpy as np


def vad_energia(audio, sample_rate=16000, frame_ms=30, threshold=0.01):
    """VAD basado en energia por frame.
    Devuelve lista de (is_speech, start_sample, end_sample)."""
    frame_size = int(frame_ms * sample_rate / 1000)
    segmentos = []
    in_segmento = False
    start = 0
    for i in range(0, len(audio), frame_size):
        frame = audio[i:i + frame_size]
        if len(frame) == 0:
            break
        rms = float(np.sqrt(np.mean(frame ** 2)))
        is_speech = rms > threshold
        if is_speech and not in_segmento:
            start = i
            in_segmento = True
        elif not is_speech and in_segmento:
            segmentos.append((True, start, i))
            in_segmento = False
    if in_segmento:
        segmentos.append((True, start, len(audio)))
    return segmentos


def vad_silero_mock(audio, sample_rate=16000, threshold=0.5):
    """Mock: silero VAD (silence / speech probabilities)."""
    if len(audio) == 0:
        return []
    frame_ms = 30
    frame_size = int(frame_ms * sample_rate / 1000)
    segmentos = []
    for i in range(0, len(audio), frame_size):
        frame = audio[i:i + frame_size]
        if len(frame) < frame_size:
            break
        rms = float(np.sqrt(np.mean(frame ** 2)))
        is_speech = rms > 0.01
        if is_speech:
            segmentos.append((True, i, i + frame_size))
    return segmentos


def endpointing_por_silencio(segmentos, min_silence_ms=500, sample_rate=16000):
    """Endpointing: detecta fin de utterance por silencio sostenido.
    Une segmentos con gaps < min_silence_ms."""
    if not segmentos:
        return []
    merged = [list(segmentos[0])]
    min_gap = int(min_silence_ms * sample_rate / 1000)
    for seg in segmentos[1:]:
        is_speech, start, end = seg
        if start - merged[-1][2] < min_gap:
            merged[-1][2] = end
        else:
            merged.append([is_speech, start, end])
    return [tuple(s) for s in merged]


def main() -> int:
    sr = 16000
    # Audio con habla + silencio + habla
    audio = np.concatenate([
        np.random.default_rng(0).normal(scale=0.1, size=sr),  # habla
        np.zeros(sr // 2),  # silencio
        np.random.default_rng(1).normal(scale=0.1, size=sr),  # habla
    ])
    segs = vad_energia(audio, sample_rate=sr, threshold=0.01)
    print(f"Segmentos detectados: {len(segs)}")
    merged = endpointing_por_silencio(segs, min_silence_ms=500, sample_rate=sr)
    print(f"Tras endpointing: {len(merged)}")
    for s, start, end in merged:
        print(f"  {start/sr:.2f}s - {end/sr:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())