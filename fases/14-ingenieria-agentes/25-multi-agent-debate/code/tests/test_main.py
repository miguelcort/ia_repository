"""Pruebas para 25-multi-agent-debate."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestDebater(unittest.TestCase):
    def test_basic(self):
        d = main.Debater("Alice", "pro")
        self.assertEqual(d.name, "Alice")
        self.assertEqual(d.stance, "pro")

    def test_argue(self):
        d = main.Debater("Alice", "pro")
        arg = d.argue("AI is good")
        self.assertIn("Alice", arg)
        self.assertEqual(len(d.arguments), 1)

    def test_argue_with_others(self):
        d = main.Debater("Alice", "pro")
        arg = d.argue("AI is good", other_arguments=["other arg"])
        self.assertIn("Counter", arg)


class TestJudge(unittest.TestCase):
    def test_basic(self):
        j = main.Judge()
        winner = j.decide(["short", "longer argument", "x"])
        self.assertEqual(winner, "longer argument")

    def test_empty(self):
        j = main.Judge()
        self.assertIsNone(j.decide([]))


class TestMultiAgentDebate(unittest.TestCase):
    def test_basic(self):
        debaters = [main.Debater("A", "pro"), main.Debater("B", "con")]
        judge = main.Judge()
        result = main.multi_agent_debate("topic", debaters, judge, max_rounds=2)
        # 2 rounds * 2 debaters = 4 args
        self.assertEqual(len(result["arguments"]), 4)
        self.assertEqual(result["rounds"], 2)
        self.assertIsNotNone(result["winner"])

    def test_history(self):
        debaters = [main.Debater("A", "pro")]
        judge = main.Judge()
        main.multi_agent_debate("topic", debaters, judge, max_rounds=3)
        self.assertEqual(len(debaters[0].arguments), 3)


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