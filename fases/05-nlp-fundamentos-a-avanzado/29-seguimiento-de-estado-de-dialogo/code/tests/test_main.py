"""Pruebas para 29-seguimiento-de-estado-de-dialogo."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestDialogueState(unittest.TestCase):
    def test_intent_reservar(self):
        ds = main.DialogueState()
        ds.update("Quiero reservar una mesa")
        self.assertEqual(ds.intent, "reservar")

    def test_intent_saludo(self):
        ds = main.DialogueState()
        ds.update("Hola")
        self.assertEqual(ds.intent, "saludo")

    def test_slot_fecha(self):
        ds = main.DialogueState()
        ds.update("fecha: 2024-12-25")
        self.assertEqual(ds.slots["fecha"], "2024-12-25")

    def test_completo(self):
        ds = main.DialogueState()
        ds.update("fecha: 2024-12-25 hora: 19:00 personas: 4 nombre: Maria")
        self.assertTrue(ds.is_complete())

    def test_incompleto(self):
        ds = main.DialogueState()
        ds.update("fecha: 2024-12-25")
        self.assertFalse(ds.is_complete())
        self.assertIn("hora", ds.missing_slots())

    def test_history(self):
        ds = main.DialogueState()
        ds.update("hola")
        ds.update("quiero reservar")
        self.assertEqual(len(ds.history), 2)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Completo", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()