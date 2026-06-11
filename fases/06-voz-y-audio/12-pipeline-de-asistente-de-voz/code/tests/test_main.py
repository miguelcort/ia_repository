"""Pruebas para 12-pipeline-de-asistente-de-voz."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestPipeline(unittest.TestCase):
    def test_basico(self):
        audio, texto = main.pipeline_asistente_voz(duracion=2)
        self.assertIsNotNone(audio)
        self.assertIn("Echo", texto)


class TestComponentes(unittest.TestCase):
    def test_captura(self):
        audio = main.capturar_audio(duracion=1, sample_rate=16000)
        self.assertEqual(len(audio), 16000)

    def test_vad(self):
        audio = np.zeros(16000)
        self.assertEqual(main.vad_silero_mock(audio), 0)

    def test_asr(self):
        audio = np.random.default_rng(0).normal(size=16000)
        texto = main.asr_whisper(audio)
        self.assertIsInstance(texto, str)

    def test_llm(self):
        resp = main.llm_respuesta("hola")
        self.assertIn("hola", resp)

    def test_tts(self):
        audio = main.tts_sintetizar("hola")
        self.assertGreater(len(audio), 0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Texto", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()