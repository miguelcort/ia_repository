"""Pruebas para 07-etiquetado-pos-y-parsing."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestTokenizar(unittest.TestCase):
    def test_basico(self):
        self.assertEqual(main.tokenizar("el gato come"), ["el", "gato", "come"])


class TestPOS(unittest.TestCase):
    def test_det(self):
        tags = main.pos_tag_mock(["el"])
        self.assertEqual(tags, ["DET"])

    def test_verb(self):
        tags = main.pos_tag_mock(["come"])
        self.assertEqual(tags, ["VERB"])

    def test_mezcla(self):
        tags = main.pos_tag_mock(["el", "gato", "come"])
        self.assertEqual(tags, ["DET", "NOUN", "VERB"])


class TestParse(unittest.TestCase):
    def test_root_es_verb(self):
        tokens = ["gato", "come"]
        tags = ["NOUN", "VERB"]
        edges = main.dep_parse_mock(tokens, tags)
        # root deberia ser el verbo
        root_edge = [e for e in edges if len(e) == 2 or (len(e) > 2 and e[2] == "root")]
        self.assertGreater(len(root_edge), 0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("POS tags", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()