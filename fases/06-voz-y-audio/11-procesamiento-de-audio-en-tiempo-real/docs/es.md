# Procesamiento de audio en tiempo real

> Procesar audio mientras se captura/reproduce. Latency < 300ms para voice assistants, < 100ms para AEC. Frontend: AEC + NS + AGC + VAD (WebRTC, RNNoise, silero). Streaming ASR: Vosk, faster-whisper. Frameworks: LiveKit, Daily, pipecat, OpenAI Realtime.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 10-modelos-de-lenguaje-de-audio
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar VAD mock.
- Implementar streaming ASR con chunks.
- Calcular latency total.
- Diagnosticar AEC + NS + AGC + VAD pipeline.

## Constrúyelo

```python
def streaming_asr_chunk(audio, sample_rate=16000, chunk_seconds=0.5):
    chunk_samples = int(chunk_seconds * sample_rate)
    resultados = []
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        n_words = max(1, len(chunk) // (sample_rate // 2))
        resultados.append(" ".join(["palabra"] * n_words))
    return resultados
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-realtime-audio
fase: 06
leccion: 11
---

1. Voice assistant: WebRTC AEC+NS+AGC+VAD+ASR+TTS.
2. Live captions: streaming ASR (Vosk, faster-whisper).
3. Conferencing: WebRTC AEC + DeepFilterNet.
4. Real-time voice: LiveKit, pipecat, OpenAI Realtime.
5. Edge: Whisper streaming, llama.cpp.
6. Latency p99 < 500ms; VAD endpointing.
```

## Ejercicios

1. **Vosk streaming**: implementar ASR streaming con
   Vosk.
2. **WebRTC frontend**: integrar AEC + NS + VAD en
   pipeline.
3. **Desafio**: voice agent real-time con LiveKit +
   faster-whisper streaming + VAD + TTS streaming.
   Latency p99 < 500ms.

## Lecturas recomendadas

- "WebRTC Audio Processing": <https://webrtc.org/>
- "Vosk": <https://alphacephei.com/vosk/>
- LiveKit: <https://livekit.io/>
- pipecat: <https://github.com/pipecat-ai/pipecat>

---

> 📚 **Adaptación al español** de la lección "[Real-time Audio Processing]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).