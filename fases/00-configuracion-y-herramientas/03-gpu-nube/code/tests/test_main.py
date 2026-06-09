"""Pruebas para el selector de backend GPU."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402

sys.path[:] = RAIZ


class TestDetectarCpu(unittest.TestCase):
    def test_cpu_siempre_disponible(self):
        info = main.detectar_cpu()
        self.assertEqual(info["backend"], "cpu")
        self.assertIn("dispositivo", info)
        self.assertEqual(info["cantidad"], 1)

    def test_cpu_devuelve_string_no_vacio(self):
        info = main.detectar_cpu()
        self.assertIsInstance(info["dispositivo"], str)
        self.assertGreater(len(info["dispositivo"]), 0)


class TestDetectarCuda(unittest.TestCase):
    def test_devuelve_none_o_dict_valido(self):
        info = main.detectar_cuda()
        if info is not None:
            self.assertEqual(info["backend"], "cuda")
            self.assertIn(info["version_cuda"], info)

    def test_cantidad_es_int_si_disponible(self):
        info = main.detectar_cuda()
        if info is not None:
            self.assertIsInstance(info["cantidad"], int)
            self.assertGreaterEqual(info["cantidad"], 1)


class TestDetectarMps(unittest.TestCase):
    def test_devuelve_none_o_dict_valido(self):
        info = main.detectar_mps()
        if info is not None:
            self.assertEqual(info["backend"], "mps")
            self.assertEqual(info["cantidad"], 1)


class TestSeleccionarBackend(unittest.TestCase):
    def test_siempre_devuelve_algo(self):
        info = main.seleccionar_backend()
        self.assertIn("backend", info)
        self.assertIn(info["backend"], {"cuda", "mps", "cpu"})

    def test_incluye_recomendacion(self):
        info = main.seleccionar_backend()
        self.assertIn("recomendacion", info)
        self.assertIsInstance(info["recomendacion"], str)
        self.assertGreater(len(info["recomendacion"]), 0)

    def test_recomendacion_segun_backend(self):
        info = main.seleccionar_backend()
        if info["backend"] == "cuda":
            self.assertIn("GPU NVIDIA", info["recomendacion"])
        elif info["backend"] == "mps":
            self.assertIn("Apple Silicon", info["recomendacion"])
        else:
            self.assertIn("Colab", info["recomendacion"])


class TestMain(unittest.TestCase):
    def test_main_imprime_json(self):
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        salida = buffer.getvalue()
        import json
        cargado = json.loads(salida)
        self.assertIn("backend", cargado)


if __name__ == "__main__":
    unittest.main()
