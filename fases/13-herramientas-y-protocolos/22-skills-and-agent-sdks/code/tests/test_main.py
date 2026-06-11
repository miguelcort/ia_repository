"""Pruebas para 22-skills-and-agent-sdks."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestSkill(unittest.TestCase):
    def test_run(self):
        s = main.make_skill("add", "add", lambda a, b: a + b)
        self.assertEqual(s.run(a=2, b=3), 5)

    def test_inputs_outputs(self):
        s = main.make_skill("foo", "foo", lambda: 1, inputs=["x"], outputs=["y"])
        self.assertEqual(s.inputs, ["x"])
        self.assertEqual(s.outputs, ["y"])


class TestAgentSDKBundle(unittest.TestCase):
    def setUp(self):
        self.sdk = main.AgentSDKBundle("test")

    def test_add_skill(self):
        s = main.make_skill("foo", "foo", lambda: 1)
        self.sdk.add_skill(s)
        self.assertTrue(self.sdk.has_skill("foo"))

    def test_run_skill(self):
        s = main.make_skill("add", "add", lambda a, b: a + b)
        self.sdk.add_skill(s)
        self.assertEqual(self.sdk.run_skill("add", a=2, b=3), 5)

    def test_run_unknown(self):
        with self.assertRaises(ValueError):
            self.sdk.run_skill("missing")

    def test_list_skills(self):
        self.sdk.add_skill(main.make_skill("a", "a", lambda: 1))
        self.sdk.add_skill(main.make_skill("b", "b", lambda: 2))
        self.assertEqual(len(self.sdk.list_skills()), 2)


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