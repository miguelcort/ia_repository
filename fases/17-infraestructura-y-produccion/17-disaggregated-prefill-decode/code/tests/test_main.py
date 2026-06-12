"""Pruebas para 17-disaggregated-prefill-decode."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestPrefill(unittest.TestCase):
    def test_create(self):
        p = main.PrefillEngine("A100", 312)
        self.assertEqual(p.compute_tflops, 312)
        self.assertFalse(p.busy)

    def test_prefill(self):
        p = main.PrefillEngine("A100", 312)
        kv = p.prefill("hello world")
        self.assertEqual(kv["prompt_len"], 11)


class TestDecode(unittest.TestCase):
    def test_create(self):
        d = main.DecodeEngine("H100", 80)
        self.assertEqual(d.memory_gb, 80)

    def test_decode_step(self):
        d = main.DecodeEngine("H100", 80)
        step = d.decode_step({"prompt_len": 5})
        self.assertEqual(step["next_token"], "x")


class TestDisagg(unittest.TestCase):
    def test_generate(self):
        e = main.DisaggregatedEngine(
            main.PrefillEngine("A100", 312),
            main.DecodeEngine("H100", 80),
        )
        result = e.generate("hello", max_tokens=3)
        self.assertEqual(len(result), 3)


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