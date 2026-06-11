"""Pruebas para 17-chatbots-de-reglas-a-neuronal."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestReglas(unittest.TestCase):
    def test_saludo(self):
        self.assertIn("Hola", main.chatbot_reglas("saludo", "Hola!"))

    def test_despedida(self):
        self.assertIn("Hasta", main.chatbot_reglas("despedida", "Chao"))

    def test_nombre(self):
        out = main.chatbot_reglas("nombre", "Me llamo Ana")
        self.assertIn("Ana", out)

    def test_no_match(self):
        self.assertIsNone(main.chatbot_reglas("otro", "xyz"))


class TestRetrieval(unittest.TestCase):
    def test_basico(self):
        faq = {"cual es el horario": "L-V 9-18"}
        r, score = main.chatbot_retrieval("cual es el horario", faq)
        self.assertEqual(r, "L-V 9-18")
        self.assertEqual(score, 1.0)


class TestGenerativo(unittest.TestCase):
    def test_genera(self):
        out = main.chatbot_generativo_mock("hola")
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 10)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Hola", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()