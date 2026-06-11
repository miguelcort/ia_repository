"""Pruebas para 18-nlp-multilingue."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestLangDetect(unittest.TestCase):
    def test_espanol(self):
        lang, _ = main.langdetect_mock("Hola, ¿cómo estás niño?")
        self.assertEqual(lang, "es")

    def test_frances(self):
        lang, _ = main.langdetect_mock("Bonjour, ça va très bien?")
        self.assertEqual(lang, "fr")

    def test_aleman(self):
        lang, _ = main.langdetect_mock("Guten Tag, schöne Straße")
        self.assertEqual(lang, "de")


class TestAlign(unittest.TestCase):
    def test_identica(self):
        pares = main.align_palabras(["a", "b", "c"], ["a", "b", "c"])
        self.assertEqual(pares, [(0, 0), (1, 1), (2, 2)])

    def test_parcial(self):
        pares = main.align_palabras(["a", "b"], ["a", "x", "b"])
        # LCS = "a" "b"
        self.assertEqual(pares, [(0, 0), (1, 2)])

    def test_vacio(self):
        pares = main.align_palabras(["a", "b"], ["x", "y"])
        self.assertEqual(pares, [])


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Alineacion", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()