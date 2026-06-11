"""Pruebas para 06-reconocimiento-de-entidades."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestTags(unittest.TestCase):
    def test_bio_basico(self):
        tokens = ["Maria", "trabaja", "en", "Google"]
        tags = main.bio_tags(tokens)
        self.assertEqual(tags, ["B-PER", "O", "O", "B-ORG"])


class TestEntidades(unittest.TestCase):
    def test_extraer(self):
        tokens = ["Maria", "Google", "Madrid"]
        tags = ["B-PER", "B-ORG", "B-LOC"]
        ents = main.extraer_entidades(tokens, tags)
        self.assertEqual(len(ents), 3)
        tipos = [e[0] for e in ents]
        self.assertIn("PER", tipos)
        self.assertIn("ORG", tipos)
        self.assertIn("LOC", tipos)

    def test_multi_token(self):
        tokens = ["New", "York", "City"]
        tags = ["B-LOC", "I-LOC", "I-LOC"]
        ents = main.extraer_entidades(tokens, tags)
        self.assertEqual(len(ents), 1)
        self.assertEqual(ents[0][0], "LOC")
        self.assertEqual(ents[0][2], "New York City")


class TestF1(unittest.TestCase):
    def test_f1_perfecto(self):
        pred = [("PER", (0, 0), "Maria")]
        gold = [("PER", (0, 0), "Maria")]
        p, r, f1 = main.span_f1(pred, gold)
        self.assertEqual(f1, 1.0)

    def test_f1_cero(self):
        pred = [("PER", (0, 0), "Maria")]
        gold = [("LOC", (0, 0), "Madrid")]
        p, r, f1 = main.span_f1(pred, gold)
        self.assertEqual(f1, 0.0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Entidades", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()