"""Pruebas para 07-bayes-y-pensamiento-estadistico."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402

sys.path[:] = RAIZ


class TestBayes(unittest.TestCase):
    def test_bayes_clasico(self):
        self.assertAlmostEqual(main.bayes(0.5, 0.9, 0.1), 0.9, places=4)

    def test_bayes_extremo(self):
        self.assertEqual(main.bayes(0.0, 0.9, 0.1), 0.0)
        self.assertEqual(main.bayes(1.0, 0.9, 0.1), 1.0)

    def test_actualizar_tras_dos_tests(self):
        p1 = main.bayes(0.01, 0.99, 0.05)
        p2 = main.actualizar(p1, 0.99, 0.05)
        self.assertGreater(p2, p1)


class TestProblemaDiagnostico(unittest.TestCase):
    def test_prevalencia_baja_sorpresa(self):
        p = main.problema_diagnostico(0.99, 0.05, 0.01)
        self.assertLess(p, 0.20)

    def test_prevalencia_alta(self):
        p = main.problema_diagnostico(0.99, 0.05, 0.5)
        self.assertGreater(p, 0.9)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("enfermedad", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
