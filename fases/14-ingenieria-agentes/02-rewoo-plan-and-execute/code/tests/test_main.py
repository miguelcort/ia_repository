"""Pruebas para 02-rewoo-plan-and-execute."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestParsePlan(unittest.TestCase):
    def test_basic(self):
        plan = "#E1 = LLM[What is the capital of France?]\n#E2 = Search[Paris]"
        tasks = main.parse_rewoo_plan(plan)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["id"], 1)
        self.assertEqual(tasks[0]["tool"], "LLM")
        self.assertEqual(tasks[1]["id"], 2)

    def test_dependencies(self):
        plan = "#E1 = LLM[hi]\n#E2 = Search[#E1]"
        tasks = main.parse_rewoo_plan(plan)
        self.assertEqual(tasks[1]["deps"], [1])

    def test_no_deps(self):
        plan = "#E1 = LLM[hi]"
        tasks = main.parse_rewoo_plan(plan)
        self.assertEqual(tasks[0]["deps"], [])


class TestDAG(unittest.TestCase):
    def test_basic(self):
        plan = "#E1 = LLM[a]\n#E2 = LLM[#E1]\n#E3 = LLM[#E2]"
        tasks = main.parse_rewoo_plan(plan)
        dag = main.build_dag(tasks)
        self.assertEqual(dag[1], [2])
        self.assertEqual(dag[2], [3])


class TestTopologicalOrder(unittest.TestCase):
    def test_basic(self):
        plan = "#E1 = LLM[a]\n#E2 = LLM[#E1]\n#E3 = LLM[#E2]"
        tasks = main.parse_rewoo_plan(plan)
        order = main.topological_order(tasks)
        # E1 must come before E2 before E3
        self.assertLess(order.index(1), order.index(2))
        self.assertLess(order.index(2), order.index(3))

    def test_independent(self):
        plan = "#E1 = LLM[a]\n#E2 = LLM[b]"
        tasks = main.parse_rewoo_plan(plan)
        order = main.topological_order(tasks)
        # both must be present
        self.assertEqual(set(order), {1, 2})


class TestExecute(unittest.TestCase):
    def test_basic(self):
        plan = "#E1 = add[1+2]\n#E2 = add[#E1+3]"
        tasks = main.parse_rewoo_plan(plan)
        registry = {"add": lambda x: eval(x)}
        results = main.execute_plan(tasks, registry)
        # E1 = 3, E2 = 6
        self.assertEqual(results[1], 3)
        self.assertEqual(results[2], 6)

    def test_unknown_tool(self):
        plan = "#E1 = missing[x]"
        tasks = main.parse_rewoo_plan(plan)
        results = main.execute_plan(tasks, {})
        self.assertIn("unknown tool", str(results[1]))


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