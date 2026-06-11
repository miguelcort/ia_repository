"""Pruebas para 23-estrategias-de-chunking-y-rag."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestChunkingFijo(unittest.TestCase):
    def test_basico(self):
        texto = "uno dos tres cuatro cinco seis siete ocho nueve diez once doce"
        chunks = main.chunking_fijo(texto, chunk_size=4, overlap=1)
        self.assertEqual(len(chunks), 4)  # [0-4], [3-7], [6-10], [9-12]
        self.assertEqual(chunks[0], "uno dos tres cuatro")

    def test_overlap(self):
        texto = "a b c d e f g h i j"
        chunks = main.chunking_fijo(texto, chunk_size=4, overlap=2)
        # Los chunks deben compartir 2 palabras
        ov = main.overlap_palabras(chunks[0], chunks[1])
        self.assertGreater(ov, 0.0)


class TestChunkingOr(unittest.TestCase):
    def test_basico(self):
        texto = "Primera oracion. Segunda oracion. Tercera oracion larga."
        chunks = main.chunking_oraciones(texto, max_chars=30)
        self.assertGreater(len(chunks), 0)


class TestChunkingSemantico(unittest.TestCase):
    def test_basico(self):
        texto = "uno dos tres cuatro cinco seis"
        chunks = main.chunking_semantico_mock(texto, n_chunks=2)
        self.assertEqual(len(chunks), 2)


class TestOverlap(unittest.TestCase):
    def test_vacio(self):
        self.assertEqual(main.overlap_palabras("", ""), 0.0)

    def test_identico(self):
        self.assertEqual(main.overlap_palabras("a b c", "a b c"), 1.0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Fijos", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()