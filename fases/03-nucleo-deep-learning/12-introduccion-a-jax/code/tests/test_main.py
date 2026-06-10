"""Pruebas para 12-introduccion-a-jax."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        salida = buffer.getvalue()
        # Acepta tanto ejecucion con JAX como mensaje sin JAX
        self.assertTrue("JAX" in salida or "grad" in salida)


if __name__ == "__main__":
    unittest.main()