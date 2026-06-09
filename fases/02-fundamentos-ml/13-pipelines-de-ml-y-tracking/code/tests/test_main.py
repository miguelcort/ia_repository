"""Pruebas para 13-pipelines-de-ml-y-tracking."""
from __future__ import annotations
import sys
import unittest
import json
import tempfile
import os
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestEstandarizar(unittest.TestCase):
    def test_media_cero(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Xt, estado = main.estandarizar(X)
        self.assertAlmostEqual(Xt.mean(axis=0)[0], 0.0, places=10)
        self.assertAlmostEqual(Xt.mean(axis=0)[1], 0.0, places=10)

    def test_std_uno(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Xt, _ = main.estandarizar(X)
        self.assertAlmostEqual(Xt.std(axis=0)[0], 1.0, places=10)


class TestImputar(unittest.TestCase):
    def test_sin_nan(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, 6.0]])
        Xt, _ = main.imputar_media(X)
        self.assertFalse(np.isnan(Xt).any())
        # 1, 3 -> media 2
        self.assertAlmostEqual(Xt[2, 0], 2.0)


class TestPipeline(unittest.TestCase):
    def test_fit_transform(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, 6.0]])
        p = main.Pipeline([
            main.PipelineStep("imputar", main.imputar_media),
            main.PipelineStep("estandarizar", main.estandarizar),
        ])
        Xt = p.fit_transform(X)
        self.assertEqual(Xt.shape, X.shape)
        self.assertFalse(np.isnan(Xt).any())

    def test_transform_replica(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        p = main.Pipeline([main.PipelineStep("estandarizar", main.estandarizar)])
        p.fit_transform(X)
        X_new = np.array([[2.0, 3.0]])
        Xt = p.transform(X_new)
        # X[:, 0] = [1, 3, 5], media = 3, std poblacional = sqrt(8/3)
        # (2 - 3) / sqrt(8/3) = -1 / 1.6329...
        media = 3.0
        std_pob = np.sqrt(8.0 / 3.0)
        esperado = (2.0 - media) / std_pob
        self.assertAlmostEqual(Xt[0, 0], esperado, places=10)


class TestRunId(unittest.TestCase):
    def test_mismo_hash(self):
        params = {"a": 1, "b": 2}
        r1 = main.run_id(params, "x")
        r2 = main.run_id(params, "x")
        self.assertEqual(r1, r2)

    def test_distinto_hash(self):
        params = {"a": 1}
        r1 = main.run_id(params, "x")
        r2 = main.run_id(params, "y")
        self.assertNotEqual(r1, r2)


class TestLogRun(unittest.TestCase):
    def test_log_crea_archivo(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            path = tf.name
        try:
            main.log_run(path, "abc123", {"lr": 0.01}, 0.95, 1.2)
            with open(path) as f:
                line = f.readline()
            entrada = json.loads(line)
            self.assertEqual(entrada["run_id"], "abc123")
            self.assertEqual(entrada["score"], 0.95)
        finally:
            os.unlink(path)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("transformado", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()