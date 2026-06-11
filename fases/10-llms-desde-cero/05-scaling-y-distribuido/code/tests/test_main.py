"""Pruebas para 05-scaling-y-distribuido."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestDataParallel(unittest.TestCase):
    def test_effective_batch(self):
        self.assertEqual(main.data_parallel_size(8, 32), 256)


class TestTensorParallel(unittest.TestCase):
    def test_layers(self):
        self.assertEqual(main.tensor_parallel_layers(32, 4), 8)


class TestPipelineParallel(unittest.TestCase):
    def test_stages(self):
        self.assertEqual(main.pipeline_parallel_stages(32, 4), 8)


class TestFSDP(unittest.TestCase):
    def test_full_shard(self):
        c = main.FSDPConfig(n_gpus=8, model_params_gb=7,
                              optimizer_states_gb=14, grad_gb=7, sharding_strategy="full")
        # Total: 28 GB, dividido por 8 = 3.5
        self.assertAlmostEqual(c.memory_per_gpu(), 3.5, places=2)

    def test_no_shard(self):
        c = main.FSDPConfig(n_gpus=8, model_params_gb=7,
                              optimizer_states_gb=14, grad_gb=7, sharding_strategy="no_shard")
        # Total: 28 GB
        self.assertAlmostEqual(c.memory_per_gpu(), 28.0, places=2)


class TestZeroStages(unittest.TestCase):
    def test_stages(self):
        s = main.zero_stages_summary()
        self.assertIn("ZeRO-1", s)
        self.assertIn("ZeRO-3", s)
        self.assertIn("FSDP", s)


class Test3DParallel(unittest.TestCase):
    def test_basic(self):
        p = main.parallelism_3d(64, tp_degree=8, pp_degree=4)
        # DP = 64 / 32 = 2
        self.assertEqual(p["dp_degree"], 2)

    def test_invalid(self):
        p = main.parallelism_3d(8, tp_degree=8, pp_degree=2)
        # 16 > 8, invalid
        self.assertIsNone(p)


class TestComm(unittest.TestCase):
    def test_single_gpu(self):
        self.assertEqual(main.communication_overhead(1, 7.0), 0.0)

    def test_multi_gpu(self):
        oh = main.communication_overhead(8, 7.0, network_gbps=200)
        self.assertGreater(oh, 0)


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