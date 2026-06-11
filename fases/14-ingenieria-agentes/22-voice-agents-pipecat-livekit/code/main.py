"""
Lección: 22-voice-agents-pipecat-livekit
Fase: 14
Voice agents: speech-to-speech en tiempo real.
Pipecat (Daily 2024), LiveKit Agents, Vocode.
+Voice +Real-time +STT/TTS.
"""
from __future__ import annotations
import time


class VoiceAgent:
    """Mock voice agent."""
    def __init__(self, name, stt="whisper", tts="elevenlabs", llm="gpt-4o"):
        self.name = name
        self.stt = stt
        self.tts = tts
        self.llm = llm
        self.conversation = []
        self.last_latency = 0

    def listen(self, audio):
        """STT: audio -> text."""
        # mock: synthesize text from audio
        text = f"transcribed from {len(audio)} samples"
        return text

    def think(self, text):
        """LLM: text -> response."""
        response = f"response to: {text[:30]}"
        return response

    def speak(self, text):
        """TTS: text -> audio."""
        # mock: synthesize audio
        audio = f"audio({len(text)} chars)"
        return audio

    def run_turn(self, audio, latency_target_ms=500):
        """Run single voice turn."""
        t0 = time.time()
        text = self.listen(audio)
        response = self.think(text)
        out_audio = self.speak(response)
        latency = (time.time() - t0) * 1000
        self.last_latency = latency
        self.conversation.append({"input": text, "response": response, "out_audio": out_audio, "latency_ms": latency})
        return out_audio

    def get_latency(self):
        return self.last_latency


def main() -> int:
    agent = VoiceAgent("voice-bot", stt="whisper", tts="elevenlabs", llm="gpt-4o")
    audio = "mock_audio_samples_1000"
    out = agent.run_turn(audio)
    print(f"Output: {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())