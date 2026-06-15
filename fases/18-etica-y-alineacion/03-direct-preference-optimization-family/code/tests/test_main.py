"""Pruebas para 03-direct-preference-optimization-family."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestF03_Direct_Preference(unittest.TestCase):
    """Tests para 03-direct-preference-optimization-family."""

    def test_module_imports(self):
        """El módulo importa correctamente."""
        self.assertTrue(hasattr(main, "main"))

    def test_funciones_definidas(self):
        """Las funciones principales existen."""
        for f in ['sigmoid', 'log_prob', 'dpo_loss', 'ipo_loss', 'kto_loss', 'orpo_loss', 'simpo_loss']:
            self.assertTrue(callable(getattr(main, f, None)),
                          f"Falta {f}")

    def test_main_ejecuta(self):
        """main() retorna 0."""
        from contextlib import redirect_stdout
        import io
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)

    def test_docstring(self):
        """El módulo tiene docstring."""
        self.assertIsNotNone(main.__doc__)

    def test_python_version(self):
        """Python version 3.10+."""
        self.assertGreaterEqual(sys.version_info, (3, 10))


if __name__ == "__main__":
    unittest.main()
