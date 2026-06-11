"""Pruebas para 05-analisis-de-sentimiento."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestLexicon(unittest.TestCase):
    def test_positivo(self):
        lex = main.LexiconSentiment()
        s = lex.predecir("excelente producto")
        self.assertEqual(s, "positivo")

    def test_negativo(self):
        lex = main.LexiconSentiment()
        s = lex.predecir("horrible servicio")
        self.assertEqual(s, "negativo")

    def test_negacion(self):
        lex = main.LexiconSentiment()
        s_no = lex.score_text("no excelente")
        s_si = lex.score_text("excelente")
        # La negacion debe cambiar el signo
        self.assertLess(s_no, s_si)


class TestClasificador(unittest.TestCase):
    def test_fit_predict(self):
        X = ["excelente", "muy malo", "genial", "horrible"]
        y = ["positivo", "negativo", "positivo", "negativo"]
        clf = main.SentimentClasificador()
        clf.fit(X, y, epocas=500)
        preds = clf.predecir(X)
        # Accuracy perfecta en train
        self.assertEqual(preds, y)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Accuracy", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()