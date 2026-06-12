"""Pruebas para 15-batch-apis."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestBatchRequest(unittest.TestCase):
    def test_create(self):
        r = main.BatchRequest("r1", "POST", "/v1/chat", {"model": "gpt-4o"})
        self.assertEqual(r.custom_id, "r1")
        self.assertEqual(r.method, "POST")


class TestBatchJob(unittest.TestCase):
    def test_create(self):
        j = main.BatchJob()
        self.assertEqual(j.total(), 0)

    def test_add(self):
        j = main.BatchJob()
        j.add(main.BatchRequest("r1", "POST", "/v1/chat", {}))
        self.assertEqual(j.total(), 1)

    def test_to_jsonl(self):
        j = main.BatchJob([
            main.BatchRequest("r1", "POST", "/v1/chat", {"model": "gpt-4o"}),
            main.BatchRequest("r2", "POST", "/v1/chat", {"model": "gpt-4o-mini"}),
        ])
        jsonl = j.to_jsonl()
        self.assertIn("r1", jsonl)
        self.assertIn("r2", jsonl)
        self.assertEqual(len(jsonl.split("\n")), 2)

    def test_estimated_cost(self):
        j = main.BatchJob([main.BatchRequest(f"r{i}", "POST", "/", {}) for i in range(10)])
        cost = j.estimated_cost(per_request=0.01, discount=0.5)
        self.assertAlmostEqual(cost, 0.05, places=4)

    def test_mark_complete(self):
        j = main.BatchJob()
        j.mark_complete([{"id": "r1", "ok": True}])
        self.assertIsNotNone(j.completed_at)
        self.assertEqual(len(j.results), 1)


class TestParse(unittest.TestCase):
    def test_parse(self):
        jsonl = '{"id": "r1", "ok": true}\n{"id": "r2", "ok": false}'
        results = main.parse_jsonl_results(jsonl)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "r1")


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