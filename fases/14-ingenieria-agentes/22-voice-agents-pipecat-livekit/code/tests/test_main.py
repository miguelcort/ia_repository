"""Pruebas para 22-voice-agents-pipecat-livekit."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestVoiceAgent(unittest.TestCase):
    def test_basic(self):
        a = main.VoiceAgent("test")
        self.assertEqual(a.name, "test")
        self.assertEqual(a.stt, "whisper")

    def test_listen(self):
        a = main.VoiceAgent("test")
        text = a.listen("audio")
        self.assertIn("transcribed", text)

    def test_think(self):
        a = main.VoiceAgent("test")
        response = a.think("hello")
        self.assertIn("response", response)

    def test_speak(self):
        a = main.VoiceAgent("test")
        audio = a.speak("hello world")
        self.assertIn("audio", audio)

    def test_run_turn(self):
        a = main.VoiceAgent("test")
        out = a.run_turn("audio samples")
        self.assertIn("audio", out)
        self.assertEqual(len(a.conversation), 1)

    def test_latency(self):
        a = main.VoiceAgent("test")
        a.run_turn("audio")
        lat = a.get_latency()
        self.assertIsInstance(lat, float)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()