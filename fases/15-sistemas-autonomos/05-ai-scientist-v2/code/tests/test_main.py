"""Pruebas para 05-ai-scientist-v2."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestResearchIdea(unittest.TestCase):
    def test_basic(self):
        i = main.ResearchIdea("title", "hyp", "method")
        self.assertEqual(i.title, "title")
        self.assertIsNone(i.results)

    def test_run_experiment(self):
        i = main.ResearchIdea("t", "h", "m")
        r = i.run_experiment()
        self.assertIn("metric", r)
        self.assertIn("significant", r)


class TestAIScientistV2(unittest.TestCase):
    def setUp(self):
        self.s = main.AIScientistV2()

    def test_basic(self):
        self.assertEqual(self.s.name, "ai_scientist")

    def test_generate_idea(self):
        i = self.s.generate_idea(lambda p: f"gen({p[:10]})", "topic")
        self.assertEqual(len(self.s.ideas), 1)
        self.assertIn("gen", i.title)

    def test_run_experiment(self):
        i = self.s.generate_idea(lambda p: "t", "topic")
        r = self.s.run_experiment(i)
        self.assertIn("metric", r)
        self.assertEqual(i.results, r)

    def test_write_paper(self):
        i = self.s.generate_idea(lambda p: "title", "topic")
        results = i.run_experiment()
        paper = self.s.write_paper(i, results)
        self.assertEqual(paper["title"], "title")
        self.assertIn("abstract", paper)
        self.assertIn("results", paper)
        self.assertEqual(len(self.s.papers), 1)

    def test_research_cycle(self):
        def llm(p):
            return f"out_{p[:5]}"
        results = self.s.research_cycle(llm, "topic", n_iterations=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(len(self.s.ideas), 3)
        self.assertEqual(len(self.s.papers), 3)
        self.assertEqual(self.s.iteration, 3)


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