"""Pruebas para 01-procesamiento-de-texto."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestNormalizar(unittest.TestCase):
    def test_lowercase(self):
        self.assertEqual(main.normalizar("Hola Mundo", lowercase=True, quitar_acentos=False), "hola mundo")

    def test_quitar_acentos(self):
        out = main.normalizar("canción", lowercase=True, quitar_acentos=True)
        self.assertEqual(out, "cancion")


class TestTokenizar(unittest.TestCase):
    def test_tokens_basico(self):
        tokens = main.tokenizar("Hola, mundo!")
        self.assertEqual(tokens, ["hola", "mundo"])

    def test_sin_palabras(self):
        self.assertEqual(main.tokenizar("!!!"), [])


class TestStopwords(unittest.TestCase):
    def test_quitar(self):
        tokens = ["el", "gato", "negro", "de"]
        out = main.quitar_stopwords(tokens)
        self.assertIn("gato", out)
        self.assertNotIn("el", out)
        self.assertNotIn("de", out)


class TestStemming(unittest.TestCase):
    def test_stem_gerundio(self):
        self.assertEqual(main.stemming(["saltando"]), ["salt"])

    def test_stem_plural(self):
        self.assertEqual(main.stemming(["perros"]), ["perro"])


class TestNGramas(unittest.TestCase):
    def test_bigrama(self):
        bg = main.n_gramas(["a", "b", "c"], n=2)
        self.assertEqual(bg, [("a", "b"), ("b", "c")])

    def test_trigrama(self):
        tg = main.n_gramas(["a", "b", "c", "d"], n=3)
        self.assertEqual(len(tg), 2)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Tokens", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()