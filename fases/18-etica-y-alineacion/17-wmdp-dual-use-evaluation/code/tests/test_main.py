"""Pruebas para 17-wmdp-dual-use-evaluation."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestF17_Wmdp_Dual_Use_Eva(unittest.TestCase):
    """Tests para 17-wmdp-dual-use-evaluation."""

    def test_module_imports(self):
        """El módulo importa correctamente."""
        self.assertTrue(hasattr(main, "main"))

    def test_funciones_definidas(self):
        """Las funciones principales existen."""
        for f in ['wmdp_eval', 'wmdp_unlearn']:
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
