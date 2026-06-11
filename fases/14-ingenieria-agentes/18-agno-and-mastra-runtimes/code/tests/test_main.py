"""Pruebas para 18-agno-and-mastra-runtimes."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestAgnoAgent(unittest.TestCase):
    def test_basic(self):
        a = main.AgnoAgent("R", "Find info")
        self.assertEqual(a.name, "R")

    def test_run(self):
        a = main.AgnoAgent("R", "G")
        result = a.run("hello")
        self.assertIn("[R]", result)
        self.assertEqual(len(a.responses), 1)

    def test_reasoning(self):
        a = main.AgnoAgent("R", "G", reasoning=True)
        result = a.run("test")
        self.assertIn("Reasoning", result)

    def test_memory(self):
        m = object()  # mock memory
        a = main.AgnoAgent("R", "G", memory=m)
        result = a.run("test")
        self.assertIn("Memory", result)

    def test_knowledge(self):
        k = object()
        a = main.AgnoAgent("R", "G", knowledge=k)
        result = a.run("test")
        self.assertIn("Knowledge", result)

    def test_add_tool(self):
        a = main.AgnoAgent("R", "G")
        a.add_tool("tool1")
        self.assertIn("tool1", a.tools)


class TestMastraAgent(unittest.TestCase):
    def test_basic(self):
        m = main.MastraAgent("W", "Execute")
        self.assertEqual(m.name, "W")

    def test_execute(self):
        m = main.MastraAgent("W", "Execute")
        result = m.execute("task")
        self.assertIn("[W]", result)

    def test_workflow(self):
        m = main.MastraAgent("W", "Execute", workflows=["wf1"])
        result = m.execute("task", workflow="wf1")
        self.assertIn("workflow", result)


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