# Detección de actividad de voz y turn-taking

> VAD: silero VAD (SOTA open, 6000+ idiomas), WebRTC VAD (rapido), pyannote.audio (multi-speaker). Endpointing: silero + silence threshold 500-1500ms, modelos predictivos (Pipecat EndOfTurnDetector). Turn-taking: barge-in, full-duplex (Moshi).

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 13-codecs-de-audio-neurales
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar VAD basado en energia.
- Implementar silero VAD mock.
- Implementar endpointing con silence threshold.
- Diagnosticar silero vs WebRTC vs pyannote.

## Constrúyelo

```python
def vad_energia(audio, sample_rate=16000, frame_ms=30, threshold=0.01):
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
    return segmentos
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-vad-turntaking
fase: 06
leccion: 14
---

1. Default: silero VAD (6000+ idiomas, ONNX).
2. Rapido: WebRTC VAD.
3. Multi-speaker: pyannote.audio 3.1.
4. Voice agent: silero + EndOfTurnDetector.
5. Mobile: silero TFLite/ONNX.
6. 16kHz mono, 30ms chunks, 500-1500ms silence.
```

## Ejercicios

1. **silero VAD**: usar silero VAD para detectar habla en
   un audio.
2. **Endpointing**: implementar silero + silence
   threshold.
3. **Desafio**: pipeline VAD + ASR streaming + endpointing
   para voice agent. Latency p99 < 500ms.

## Lecturas recomendadas

- "Silero VAD" (Snakers, 2023)
- pyannote.audio: <https://github.com/pyannote/pyannote-audio>
- LiveKit turn-detection: <https://docs.livekit.io/agents/build/turns/>

---

> 📚 **Adaptación al español** de la lección "[VAD and Turn-taking]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).