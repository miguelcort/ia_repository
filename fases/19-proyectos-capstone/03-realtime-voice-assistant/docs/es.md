# 03 — Realtime voice assistant

> Realtime voice assistants (Pipecat, LiveKit, GPT-4o realtime, Moshi): full-duplex audio con latency <300ms. VAD (Silero), STT (Whisper), LLM streaming (Llama 3, GPT-4o), TTS (ElevenLabs, XTTS). Turn-taking y barge-in son los problemas difíciles.

**Tipo:** Capstone
**Lenguajes:** Python (Pipecat), TypeScript (web client)
**Prerrequisitos:** Fase 6 (voz/audio), Fase 11 (LLM), Fase 13 (tools)
**Tiempo estimado:** 25 horas

## Objetivos

- Implementar pipeline realtime: VAD + STT + LLM + TTS.
- Lograr latency end-to-end <500ms.
- Soportar barge-in (interrupción del usuario).
- Evaluar MOS (Mean Opinion Score).

## El problema

Voice assistants en 2026 (Pipecat, LiveKit Agents)
logran conversaciones full-duplex con latencia <300ms.
El pipeline clásico: VAD detecta habla, STT transcribe
streaming, LLM genera tokens streaming, TTS sintetiza
audio streaming. Barge-in (usuario interrumpe al
asistente) requiere cancelación de TTS + context
preservation. Voice activity detection (Silero VAD v5)
funciona en 32ms chunks.

## Constrúyelo

```python
import asyncio
from pipecat.frames import AudioFrame, TextFrame
from pipecat.pipeline import Pipeline
from pipecat.services.openai import OpenAILLM
from pipecat.services.elevenlabs import ElevenLabsTTS
from pipecat.vad.silero import SileroVAD


async def build_pipeline():
    vad = SileroVAD()
    stt = WhisperSTT(model="whisper-large-v3")
    llm = OpenAILLM(model="gpt-4o-realtime")
    tts = ElevenLabsTTS(voice_id="alloy")
    return Pipeline([vad, stt, llm, tts])
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-voice-assistant
fase: 19
leccion: 03
---

1. VAD (Silero, WebRTC).
2. STT streaming (Whisper).
3. LLM streaming (gpt-4o-realtime).
4. TTS streaming (ElevenLabs).
5. Barge-in handling.
```

## Ejercicios

1. **Pipecat setup**: instalar y correr
   hello-world.
2. **Barge-in**: implementar VAD-based
   interruption.
3. **Eval**: MOS test con 10 speakers.

## Lecturas recomendadas

- "Pipecat" (Daily 2024)
- "GPT-4o Realtime" (OpenAI 2024)
- "Moshi" (Kyutai 2024)
- "LiveKit Agents" (LiveKit 2024)

---

> 📚 **Adaptación al español** de la lección
> "[03-realtime-voice-assistant]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
