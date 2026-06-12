"""Pruebas para 02-fipa-acl-heritage."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestPerformatives(unittest.TestCase):
    def test_list(self):
        perfs = main.list_performatives()
        self.assertIn("inform", perfs)
        self.assertIn("request", perfs)
        self.assertIn("query", perfs)
        self.assertIn("propose", perfs)
        self.assertIn("cfp", perfs)

    def test_get(self):
        p = main.get_performative("inform")
        self.assertEqual(p["name"], "inform")
        self.assertIn("example", p)


class TestACLMessage(unittest.TestCase):
    def test_create(self):
        m = main.ACLMessage("inform", "a1", "a2", "hello")
        self.assertEqual(m.performative, "inform")
        self.assertEqual(m.sender, "a1")
        self.assertEqual(m.receiver, "a2")

    def test_to_dict(self):
        m = main.ACLMessage("request", "a1", "a2", "compute", ontology="math")
        d = m.to_dict()
        self.assertEqual(d["ontology"], "math")
        self.assertEqual(d["performative"], "request")

    def test_default_ids(self):
        m = main.ACLMessage("inform", "a1", "a2", "x")
        self.assertIsNotNone(m.conversation_id)
        self.assertIsNotNone(m.reply_with)

    def test_custom_ids(self):
        m = main.ACLMessage("inform", "a1", "a2", "x", conversation_id="c1", reply_with="r1")
        self.assertEqual(m.conversation_id, "c1")
        self.assertEqual(m.reply_with, "r1")


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()