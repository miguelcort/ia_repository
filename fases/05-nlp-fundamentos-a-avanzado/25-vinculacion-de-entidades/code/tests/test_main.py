"""Pruebas para 25-vinculacion-de-entidades."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestKB(unittest.TestCase):
    def test_get(self):
        kb = main.KnowledgeBase()
        ent = kb.get("Q76")
        self.assertEqual(ent["name"], "Barack Obama")

    def test_search(self):
        kb = main.KnowledgeBase()
        res = kb.search("Obama fue presidente")
        self.assertIn("Q76", res)


class TestLinking(unittest.TestCase):
    def test_exact(self):
        kb = main.KnowledgeBase()
        qid = main.entity_linking("Barack Obama", kb)
        self.assertEqual(qid, "Q76")

    def test_alias(self):
        kb = main.KnowledgeBase()
        qid = main.entity_linking("Obama", kb)
        self.assertEqual(qid, "Q76")

    def test_partial(self):
        kb = main.KnowledgeBase()
        qid = main.entity_linking("Berlin", kb)
        self.assertEqual(qid, "Q64")

    def test_no_match(self):
        kb = main.KnowledgeBase()
        qid = main.entity_linking("xyz123", kb)
        self.assertIsNone(qid)


class TestLinkear(unittest.TestCase):
    def test_basico(self):
        kb = main.KnowledgeBase()
        res = main.linkear_texto("Barack Obama y United States", kb)
        # Debe encontrar al menos 2 entidades
        self.assertGreaterEqual(len(res), 2)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("vinculadas", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()