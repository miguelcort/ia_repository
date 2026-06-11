"""Pruebas para 21-llm-routing-layer."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestRouteByCost(unittest.TestCase):
    def test_basic(self):
        models = [
            {"name": "a", "cost_per_1k": 5.0},
            {"name": "b", "cost_per_1k": 1.0},
            {"name": "c", "cost_per_1k": 0.5},
        ]
        m = main.route_by_cost("test", models)
        self.assertEqual(m["name"], "c")


class TestRouteByLatency(unittest.TestCase):
    def test_basic(self):
        models = [
            {"name": "a", "cost_per_1k": 1.0},
            {"name": "b", "cost_per_1k": 1.0},
        ]
        history = {"a": [0.5, 0.4, 0.6], "b": [0.1, 0.1, 0.1]}
        m = main.route_by_latency("test", models, history)
        self.assertEqual(m["name"], "b")


class TestRouteByCapability(unittest.TestCase):
    def test_found(self):
        models = [
            {"name": "a", "capabilities": ["chat"]},
            {"name": "b", "capabilities": ["code", "chat"]},
        ]
        m = main.route_by_capability("test", models, "code")
        self.assertEqual(m["name"], "b")

    def test_not_found(self):
        models = [{"name": "a", "capabilities": ["chat"]}]
        m = main.route_by_capability("test", models, "vision")
        self.assertIsNone(m)


class TestFallback(unittest.TestCase):
    def test_primary_works(self):
        models = [{"name": "a"}, {"name": "b"}]
        def primary(m, p):
            return f"ok from {m['name']}"
        r = main.fallback_chain("test", models, primary)
        self.assertEqual(r, "ok from a")

    def test_fallback(self):
        models = [{"name": "a"}, {"name": "b"}]
        def primary(m, p):
            if m["name"] == "a":
                raise RuntimeError("fail")
            return f"ok from {m['name']}"
        r = main.fallback_chain("test", models, primary)
        self.assertEqual(r, "ok from b")

    def test_all_fail(self):
        models = [{"name": "a"}]
        def primary(m, p):
            raise RuntimeError("fail")
        r = main.fallback_chain("test", models, primary)
        self.assertIsNone(r)


class TestLoadBalance(unittest.TestCase):
    def test_basic(self):
        models = [{"name": "a"}, {"name": "b"}]
        counts = main.load_balance_round_robin("test", models, n_requests=10)
        # each gets 5
        self.assertEqual(counts["a"], 5)
        self.assertEqual(counts["b"], 5)


class TestSemanticRoute(unittest.TestCase):
    def test_basic(self):
        emb = [1.0, 0.0, 0.0]
        model_embs = [
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        idx = main.semantic_route("test", emb, model_embs, threshold=0.5)
        self.assertEqual(idx, 0)

    def test_no_match(self):
        emb = [1.0, 0.0, 0.0]
        model_embs = [[0.0, 1.0, 0.0]]
        idx = main.semantic_route("test", emb, model_embs, threshold=0.9)
        self.assertIsNone(idx)


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