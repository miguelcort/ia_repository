# Voice agents Pipecat LiveKit

> Voice agents Pipecat y LiveKit. Pipecat (Daily 2024): voice agent framework con STT + TTS + LLM + real-time + production. LiveKit Agents (2024): voice + video + real-time + WebRTC + production. Voice loop: (1) listen (STT: Whisper, Deepgram), (2) think (LLM), (3) speak (TTS: ElevenLabs, OpenAI), (4) real-time <500ms latency + streaming. Providers SOTA: STT (Whisper OpenAI, Deepgram, AssemblyAI, Google STT), TTS (ElevenLabs, OpenAI TTS, Play.ht, Google TTS). +Voice, +Real-time, +STT/TTS, +Latency, +Optimize, +Production, +Reliable, +Standard, +Streaming. Variants: Pipecat seminal, LiveKit seminal, Vocode (2023 +voice +Python +open source), Daily, custom, OpenAI Voice, Anthropic Voice. Frameworks: pipecat, livekit, vocode, daily, openai, anthropic, elevenlabs, deepgram. +Production: standard 2024-25. +Use cases: voice, video, real-time, customer service, accessibility, call center. Decision: voice -> Pipecat o Vocode, video -> LiveKit, production -> Pipecat o LiveKit, simple -> custom. Trade-offs: cada uno + specialty. 2025: +MCP + A2A + native + voice.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12/19
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar VoiceAgent con STT + TTS + LLM.
- Implementar listen/think/speak.
- Implementar run_turn con latency tracking.
- Diagnosticar Pipecat vs LiveKit vs Vocode.
- Diagnosticar providers SOTA.

## Constrúyelo

```python
class VoiceAgent:
    def run_turn(self, audio, latency_target_ms=500):
        text = self.listen(audio)
        response = self.think(text)
        out_audio = self.speak(response)
        return out_audio
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: voice-agents
fase: 14
leccion: 22
---

1. STT + TTS + LLM.
2. Listen + think + speak.
3. Real-time <500ms.
4. Voice loop.
5. +Production.
```

## Ejercicios

1. **Pipecat**: usar Pipecat
   con STT + TTS.
2. **LiveKit**: probar
   LiveKit voice agent.
3. **Desafio**: real-time
   voice agent production.

## Lecturas recomendadas

- "Pipecat: Voice Agent Framework" (Daily, 2024)
- "LiveKit Agents" (LiveKit, 2024)
- "Vocode: Open Source Voice Agent" (Vocode, 2023)
- "OpenAI Voice Mode" (OpenAI, 2024)

---

> 📚 **Adaptación al español de la lección [Voice Agents Pipecat LiveKit]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).