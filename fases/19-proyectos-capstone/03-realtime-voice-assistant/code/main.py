"""
Lección: 03-realtime-voice-assistant
Fase: 19
Capstone de ingeniería AI: 03 Realtime Voice Assistant.
"""
from __future__ import annotations
import sys

def build_pipeline():
    """Build Pipecat pipeline: VAD + STT + LLM + TTS."""
    return {"vad": "silero", "stt": "whisper-large-v3",
            "llm": "gpt-4o-realtime",
            "tts": "elevenlabs-alloy"}


def barge_in_check(audio_chunk, vad):
    """Detect if user is interrupting."""
    return vad.is_speech(audio_chunk)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
