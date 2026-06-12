"""Pruebas para 18-theory-of-mind-coordination."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestMindModel(unittest.TestCase):
    def test_create(self):
        m = main.MindModel("a1")
        self.assertEqual(m.agent_id, "a1")
        self.assertEqual(m.beliefs, {})

    def test_beliefs(self):
        m = main.MindModel("a1")
        m.update_belief("k", "v")
        self.assertEqual(m.beliefs["k"], "v")

    def test_desires(self):
        m = main.MindModel("a1")
        m.add_desire("ship")
        self.assertIn("ship", m.desires)

    def test_intention(self):
        m = main.MindModel("a1")
        m.set_intention("code")
        self.assertEqual(m.predict_action(), "code")

    def test_predict_default(self):
        m = main.MindModel("a1")
        self.assertEqual(m.predict_action(), "wait")


class TestTheoryOfMind(unittest.TestCase):
    def setUp(self):
        self.tom = main.TheoryOfMind("alice")
        self.tom.self_mind.update_belief("task", "ship")
        self.tom.self_mind.update_belief("language", "python")

    def test_model_other(self):
        self.tom.model_other("bob", {"task": "ship"})
        self.assertIn("bob", self.tom.others_minds)

    def test_update_other(self):
        self.tom.update_other_belief("bob", "k", "v")
        self.assertEqual(self.tom.others_minds["bob"].beliefs["k"], "v")

    def test_update_auto_creates(self):
        self.tom.update_other_belief("charlie", "x", 1)
        self.assertIn("charlie", self.tom.others_minds)

    def test_predict_other(self):
        self.tom.model_other("bob")
        self.tom.others_minds["bob"].set_intention("test")
        self.assertEqual(self.tom.predict_other_action("bob"), "test")

    def test_predict_unknown(self):
        self.assertIsNone(self.tom.predict_other_action("unknown"))

    def test_recursive_depth(self):
        self.tom.model_other("bob", {"k": "v"})
        r = self.tom.recursive_depth("bob", depth=2)
        self.assertEqual(r["level"], 2)
        self.assertEqual(r["modeled_beliefs"]["k"], "v")

    def test_recursive_depth_zero(self):
        r = self.tom.recursive_depth("bob", depth=0)
        self.assertIsNone(r)

    def test_common_ground(self):
        self.tom.model_other("bob", {"task": "ship", "framework": "django"})
        common = self.tom.find_common_ground("bob")
        self.assertIn("task", common)
        self.assertNotIn("language", common)

    def test_common_ground_unknown(self):
        self.assertEqual(self.tom.find_common_ground("unknown"), set())


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