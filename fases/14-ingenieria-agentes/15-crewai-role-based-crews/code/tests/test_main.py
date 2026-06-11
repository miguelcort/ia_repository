"""Pruebas para 15-crewai-role-based-crews."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestCrewAIAgent(unittest.TestCase):
    def test_basic(self):
        a = main.CrewAIAgent("Researcher", "Find info", "Expert")
        self.assertEqual(a.role, "Researcher")
        self.assertEqual(a.goal, "Find info")

    def test_execute(self):
        a = main.CrewAIAgent("R", "G", "B")
        task = main.CrewAITask("Do X", a)
        output = a.execute(task)
        self.assertIn("R", output)
        self.assertEqual(len(a.outputs), 1)

    def test_to_dict(self):
        a = main.CrewAIAgent("R", "G", "B", tools=["t1", "t2"], llm="gpt-4o")
        d = a.to_dict()
        self.assertEqual(d["tools"], ["t1", "t2"])
        self.assertEqual(d["llm"], "gpt-4o")


class TestCrewAITask(unittest.TestCase):
    def test_basic(self):
        a = main.CrewAIAgent("R", "G", "B")
        t = main.CrewAITask("Do X", a, expected_output="X")
        self.assertEqual(t.expected_output, "X")
        self.assertEqual(t.context, [])

    def test_run(self):
        a = main.CrewAIAgent("R", "G", "B")
        t = main.CrewAITask("Do X", a)
        out = t.run()
        self.assertIsNotNone(out)
        self.assertEqual(t.output, out)


class TestCrew(unittest.TestCase):
    def test_sequential(self):
        a1 = main.CrewAIAgent("A", "G", "B")
        a2 = main.CrewAIAgent("B", "G", "B")
        t1 = main.CrewAITask("t1", a1)
        t2 = main.CrewAITask("t2", a2)
        crew = main.Crew([a1, a2], [t1, t2], process="sequential")
        results = crew.kickoff()
        self.assertEqual(len(results), 2)

    def test_parallel(self):
        a1 = main.CrewAIAgent("A", "G", "B")
        t1 = main.CrewAITask("t1", a1)
        crew = main.Crew([a1], [t1], process="parallel")
        results = crew.kickoff()
        self.assertEqual(len(results), 1)


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