# Streaming speech-to-speech

> Pipeline full-duplex low-latency: AEC + VAD + ASR streaming + LLM streaming + TTS streaming. Latency < 500ms. SOTA: Moshi (200ms), OpenAI Realtime (~300ms), Ultravox. Frameworks: pipecat, LiveKit Agents, OpenAI Realtime.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14-deteccion-de-actividad-de-voz-y-turn-taking
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar pipeline ASR -> LLM streaming.
- Generar tokens de TTS streaming.
- Calcular latencia full-duplex.
- Diagnosticar Moshi vs OpenAI Realtime.

## Constrúyelo

```python
def streaming_asr_to_llm_pipeline(audio_chunk, sample_rate=16000):
    if len(audio_chunk) < sample_rate // 4:
        return None
    texto = " ".join(["palabra"] * max(1, len(audio_chunk) // (sample_rate // 2)))
    return texto
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-streaming-s2s
fase: 06
leccion: 15
---

1. Max calidad: OpenAI Realtime, Gemini Live.
2. Self-host: Moshi (200ms), Ultravox.
3. Cascada: LiveKit, pipecat + faster-whisper + Claude +
   Cartesia Sonic.
4. Telefonia: Vocode, Twilio.
5. Latency p99 < 500ms; barge-in; WebSocket.
```

## Ejercicios

1. **LiveKit voice agent**: implementar voice agent
   con LiveKit Agents + Claude + faster-whisper.
2. **Moshi self-host**: correr Moshi localmente y
   medir latency.
3. **Desafio**: pipeline production con OpenAI Realtime
   + barge-in + tool calling + memory. Latency p99 <
   500ms, WER < 5%.

## Lecturas recomendadas

- "Moshi" (Kyutai, 2024)
- "OpenAI Realtime API": <https://platform.openai.com/docs/guides/realtime>
- "Ultravox" (Fixie.ai, 2024)
- LiveKit: <https://livekit.io/>

---

> 📚 **Adaptación al español** de la lección "[Streaming Speech-to-Speech]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).