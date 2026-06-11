"""Pruebas para 20-salidas-estructuradas-y-decoding-constrenido."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestForzarJson(unittest.TestCase):
    def test_basico(self):
        obj = main.forzar_json("nombre: Maria, edad: 30", ["nombre", "edad"])
        self.assertEqual(obj["nombre"], "Maria")
        self.assertEqual(obj["edad"], 30.0)


class TestEsquema(unittest.TestCase):
    def test_valido(self):
        obj = {"nombre": "Maria", "edad": 30.0}
        self.assertTrue(main.validar_esquema(obj, {"nombre": str, "edad": float}))

    def test_invalido_tipo(self):
        obj = {"nombre": 123}  # int, no str
        self.assertFalse(main.validar_esquema(obj, {"nombre": str}))

    def test_falta_clave(self):
        obj = {"nombre": "Maria"}
        self.assertFalse(main.validar_esquema(obj, {"nombre": str, "edad": float}))


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Parseado", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()