"""Pruebas para 12-depuracion-y-profiling."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402

sys.path[:] = RAIZ


class TestFuncionLenta(unittest.TestCase):
    def test_lenta_y_rapida_mismo_resultado(self):
        n = 50
        self.assertEqual(main.funcion_lenta(n), main.version_rapida(n))


class TestMicroBenchmark(unittest.TestCase):
    def test_benchmark_devuelve_campos(self):
        r = main.micro_benchmark(10, repeticiones=2)
        self.assertEqual(r["n"], 10)
        self.assertEqual(r["repeticiones"], 2)
        self.assertGreater(r["lenta_s"], 0)
        self.assertGreaterEqual(r["rapida_s"], 0)
        self.assertGreater(r["speedup"], 1)

    def test_rapida_es_mucho_mas_veloz(self):
        r = main.micro_benchmark(50, repeticiones=3)
        self.assertGreater(r["lenta_s"], r["rapida_s"])


class TestPerfilar(unittest.TestCase):
    def test_perfil_muestra_funcion_lenta(self):
        salida = main.perfilar(20, top=3)
        self.assertIn("funcion_lenta", salida)

    def test_perfil_es_string(self):
        salida = main.perfilar(10, top=2)
        self.assertIsInstance(salida, str)


class TestMemoria(unittest.TestCase):
    def test_reporte_tiene_campos(self):
        r = main.memoria(50)
        self.assertIn("bytes_actual", r)
        self.assertIn("bytes_pico", r)
        self.assertIn("kb_pico", r)

    def test_pico_es_no_negativo(self):
        r = main.memoria(10)
        self.assertGreaterEqual(r["bytes_pico"], 0)


class TestDepurar(unittest.TestCase):
    def test_depurar_devuelve_string(self):
        s = main.depurar(1)
        self.assertIsInstance(s, str)


class TestMain(unittest.TestCase):
    def test_main_benchmark(self):
        import io
        from contextlib import redirect_stdout
        import json
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main(["--modo", "benchmark", "--n", "10"])
        self.assertEqual(rc, 0)
        data = json.loads(buffer.getvalue())
        self.assertEqual(data["n"], 10)

    def test_main_memory(self):
        import io
        from contextlib import redirect_stdout
        import json
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main(["--modo", "memory", "--n", "5"])
        self.assertEqual(rc, 0)
        data = json.loads(buffer.getvalue())
        self.assertIn("bytes_pico", data)

    def test_main_profile(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main(["--modo", "profile", "--n", "5"])
        self.assertEqual(rc, 0)
        self.assertIn("funcion_lenta", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
