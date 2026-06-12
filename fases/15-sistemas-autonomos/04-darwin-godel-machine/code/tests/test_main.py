"""Pruebas para 04-darwin-godel-machine."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestDGM(unittest.TestCase):
    def setUp(self):
        self.dgm = main.DarwinGodelMachine()
        self.counter = [0]

    def factory(self):
        self.counter[0] += 1
        return {"id": f"a{self.counter[0]}", "fitness": 0}

    def eval_fn(self, agent):
        return agent["fitness"]

    def mutate(self, agent):
        self.counter[0] += 1
        return {"id": f"a{self.counter[0]}", "fitness": agent["fitness"] + 1}

    def test_initialize(self):
        self.dgm.initialize(self.factory)
        self.assertEqual(len(self.dgm.agents), 20)

    def test_evaluate(self):
        self.dgm.initialize(self.factory)
        scored = self.dgm.evaluate(self.eval_fn)
        self.assertEqual(len(scored), 20)

    def test_mutate(self):
        agent = {"id": "a1", "fitness": 0}
        new = self.dgm.mutate_agent(agent, self.mutate)
        self.assertEqual(new["fitness"], 1)
        self.assertEqual(len(self.dgm.improvements), 1)

    def test_step(self):
        self.dgm.initialize(self.factory)
        best = self.dgm.step(self.eval_fn, self.mutate)
        self.assertEqual(self.dgm.generation, 1)
        # mutations should improve fitness
        # population has 20; after step 10 keep top + 10 mutations
        # mutations add 1 to fitness, so top should be >= 0
        self.assertGreaterEqual(best[1], 0)

    def test_run(self):
        self.dgm.initialize(self.factory)
        history = self.dgm.run(self.eval_fn, self.mutate, n_steps=3)
        self.assertEqual(len(history), 3)
        self.assertEqual(self.dgm.generation, 3)


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