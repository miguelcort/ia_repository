"""Pruebas para 05-autonomous-research-agent."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class Test05_Autonomous_Research_Ag(unittest.TestCase):
    def test_module_imports(self):
        self.assertTrue(hasattr(main, "main"))

    def test_main_ejecuta(self):
        from contextlib import redirect_stdout
        import io
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)

    def test_docstring(self):
        self.assertIsNotNone(main.__doc__)

    def test_python_version(self):
        self.assertGreaterEqual(sys.version_info, (3, 10))

    def test_funciones_globales(self):
        """El módulo tiene funciones o clases."""
        attrs = [a for a in dir(main) if not a.startswith("_")]
        self.assertGreater(len(attrs), 0)


if __name__ == "__main__":
    unittest.main()
