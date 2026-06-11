# Pipeline de asistente de voz

> Voice agent: captura -> AEC + NS + VAD -> ASR streaming -> LLM (Claude/GPT-4) -> TTS streaming. Full-duplex con barge-in y turn-taking. Frameworks: pipecat, LiveKit Agents, OpenAI Realtime, Ultravox, Moshi.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11-procesamiento-de-audio-en-tiempo-real
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar pipeline captura -> VAD -> ASR -> LLM -> TTS.
- Diagnosticar barge-in y turn-taking.
- Evaluar voice agents production.
- Diagnosticar frameworks SOTA.

## Constrúyelo

```python
def pipeline_asistente_voz(duracion=5, sample_rate=16000):
    audio = capturar_audio(duracion, sample_rate)
    if vad_silero_mock(audio) == 0:
        return None, "Silencio"
    texto = asr_whisper(audio, sample_rate)
    respuesta = llm_respuesta(texto)
    return tts_sintetizar(respuesta), respuesta
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-voice-agent
fase: 06
leccion: 12
---

1. Default: pipecat, LiveKit Agents.
2. API: OpenAI Realtime, Gladia, Deepgram.
3. Self-host: Moshi, Ultravox, pipecat.
4. Baja latencia: Cartesia Sonic, Ultravox.
5. Telefonia: Vocode, Twilio, Bland AI.
6. Latency p99 < 800ms; barge-in; VAD endpointing.
```

## Ejercicios

1. **pipecat voice agent**: crear voice agent con
   pipecat + Claude + Whisper + ElevenLabs.
2. **LiveKit Agents**: implementar voice agent
   multi-agent con LiveKit.
3. **Desafio**: voice agent production con barge-in,
   interruption, memory multi-turn, tools, RAG.
   Eval latency p99 < 800ms, WER < 5%, completion > 80%.

## Lecturas recomendadas

- pipecat: <https://github.com/pipecat-ai/pipecat>
- LiveKit Agents: <https://docs.livekit.io/agents/>
- OpenAI Realtime: <https://platform.openai.com/docs/guides/realtime>

---

> 📚 **Adaptación al español** de la lección "[Voice Assistant Pipeline]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).