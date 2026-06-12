"""Pruebas para 12-edge-inference."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestEdgeDevice(unittest.TestCase):
    def test_create(self):
        d = main.EdgeDevice("phone", memory_mb=8192)
        self.assertEqual(d.memory_mb, 8192)
        self.assertIsNone(d.running_model)

    def test_can_run(self):
        d = main.EdgeDevice("phone", memory_mb=8192)
        self.assertTrue(d.can_run(4096))
        self.assertFalse(d.can_run(16384))

    def test_can_run_quantization_required(self):
        d = main.EdgeDevice("phone", supports_quantization=False)
        self.assertFalse(d.can_run(1000, requires_quantization=True))

    def test_load(self):
        d = main.EdgeDevice("phone", memory_mb=8192)
        self.assertTrue(d.load("model", 4096))
        self.assertEqual(d.running_model, "model")

    def test_load_fail(self):
        d = main.EdgeDevice("phone", memory_mb=1024)
        self.assertFalse(d.load("model", 4096))

    def test_unload(self):
        d = main.EdgeDevice("phone", memory_mb=8192)
        d.load("model", 4096)
        d.unload()
        self.assertIsNone(d.running_model)


class TestRouter(unittest.TestCase):
    def setUp(self):
        self.r = main.EdgeRouter()
        self.r.register_device(main.EdgeDevice("phone", memory_mb=8192))
        self.r.register_device(main.EdgeDevice("laptop", memory_mb=16384))
        self.r.set_cloud("cloud.example.com")

    def test_route_edge(self):
        result = self.r.route("model", 4096)
        self.assertEqual(result[0], "edge")

    def test_route_cloud(self):
        result = self.r.route("huge", 65536)
        self.assertEqual(result[0], "cloud")

    def test_route_no_capacity(self):
        r = main.EdgeRouter()
        result = r.route("m", 100)
        self.assertIsNone(result[0])


class TestEstimate(unittest.TestCase):
    def test_basic(self):
        lat = main.estimate_latency(1000, 10.0, tokens=100)
        self.assertGreater(lat, 0)

    def test_zero_compute(self):
        lat = main.estimate_latency(1000, 0.0, tokens=100)
        self.assertIsNotNone(lat)


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