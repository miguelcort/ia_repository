"""Pruebas para 26-extraccion-de-relaciones-y-grafo-de-conocimiento."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestEntidades(unittest.TestCase):
    def test_basico(self):
        ents = main.entidades_en_texto("Maria vive en Madrid. Pedro trabaja en Google.")
        self.assertIn("Maria", ents)
        self.assertIn("Madrid", ents)
        self.assertIn("Pedro", ents)
        self.assertIn("Google", ents)


class TestRelaciones(unittest.TestCase):
    def test_presidente(self):
        rels = main.extraer_relaciones_patron("Maria es presidente de Pedro")
        # Maria y Pedro ambos mayusculas
        # Patron: "Maria es presidente de Pedro" -> "presidente_de"
        self.assertGreater(len(rels), 0)


class TestClasificar(unittest.TestCase):
    def test_trabaja(self):
        rel = main.clasificar_relacion_mock("Maria", "Pedro", "Maria es presidente de Pedro")
        self.assertEqual(rel, "trabaja_para")

    def test_ubicado(self):
        rel = main.clasificar_relacion_mock("Maria", "Madrid", "Maria vive en Madrid")
        self.assertEqual(rel, "ubicado_en")


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Entidades", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()