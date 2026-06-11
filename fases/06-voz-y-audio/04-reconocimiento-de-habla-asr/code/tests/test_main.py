"""Pruebas para 04-reconocimiento-de-habla-asr."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestWER(unittest.TestCase):
    def test_perfecto(self):
        self.assertEqual(main.wer("el gato come", "el gato come"), 0.0)

    def test_eliminar(self):
        # 'pescado' es deletion
        self.assertAlmostEqual(main.wer("el gato come pescado", "el gato come"), 1 / 4)

    def test_sustituir(self):
        # 'come' -> 'bebe'
        self.assertAlmostEqual(main.wer("el gato come", "el gato bebe"), 1 / 3)

    def test_vacio(self):
        self.assertEqual(main.wer("", ""), 0.0)


class TestCER(unittest.TestCase):
    def test_basico(self):
        # Sin espacios, "elgatocome" vs "elgato"
        cer = main.cer("el gato come", "el gato")
        self.assertGreater(cer, 0.0)


class TestTokenizar(unittest.TestCase):
    def test_basico(self):
        tokens = main.tokenizar_caracteres("hola mundo")
        self.assertEqual(tokens, list("hola_mundo"))


class TestMockASR(unittest.TestCase):
    def test_decode(self):
        mel = np.random.default_rng(0).normal(size=(80, 100))
        out = main.mock_asr_decode(mel)
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("WER", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()