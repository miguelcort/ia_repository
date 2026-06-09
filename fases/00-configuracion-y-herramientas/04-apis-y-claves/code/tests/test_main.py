"""Pruebas para el cargador y validador de claves API."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402

sys.path[:] = RAIZ


class TestCargarEnv(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False, encoding="utf-8"
        )
        self.tmp.write("FOO_TEST=bar\n")
        self.tmp.write("# comentario\n")
        self.tmp.write("CON_ESPACIOS='valor con espacios'\n")
        self.tmp.write("VACIA=\n")
        self.tmp.write("\n")
        self.tmp.close()
        # Limpiar variables que el test usa
        for k in ("FOO_TEST", "CON_ESPACIOS", "VACIA"):
            os.environ.pop(k, None)

    def tearDown(self):
        Path(self.tmp.name).unlink()
        for k in ("FOO_TEST", "CON_ESPACIOS", "VACIA"):
            os.environ.pop(k, None)

    def test_carga_claves_desde_archivo(self):
        cargados = main.cargar_env(Path(self.tmp.name))
        self.assertGreaterEqual(cargados, 2)
        self.assertEqual(os.environ.get("FOO_TEST"), "bar")
        self.assertEqual(os.environ.get("CON_ESPACIOS"), "valor con espacios")

    def test_no_explota_si_archivo_no_existe(self):
        self.assertEqual(main.cargar_env(Path("/no/existe.env")), 0)

    def test_no_sobrescribe_variable_existente(self):
        os.environ["FOO_TEST"] = "existente"
        main.cargar_env(Path(self.tmp.name))
        self.assertEqual(os.environ["FOO_TEST"], "existente")


class TestValidarClave(unittest.TestCase):
    def test_vacia_es_invalida(self):
        ok, msg = main.validar_clave("X", "")
        self.assertFalse(ok)
        self.assertIn("vacia", msg)

    def test_placeholder_es_invalido(self):
        ok, _ = main.validar_clave("X", "your-key-here")
        self.assertFalse(ok)
        ok, _ = main.validar_clave("X", "changeme")
        self.assertFalse(ok)

    def test_prefijo_incorrecto_es_invalido(self):
        ok, msg = main.validar_clave("OPENAI_API_KEY", "abc1234567890abc", "sk-")
        self.assertFalse(ok)
        self.assertIn("sk-", msg)

    def test_prefijo_correcto_y_largo_valido(self):
        ok, msg = main.validar_clave("OPENAI_API_KEY", "sk-" + "x" * 50, "sk-")
        self.assertTrue(ok)
        self.assertIn("ok", msg)

    def test_clave_corta_es_invalida(self):
        ok, _ = main.validar_clave("X", "sk-corto", "sk-")
        self.assertFalse(ok)

    def test_sin_prefijo_no_falla_por_prefijo(self):
        ok, msg = main.validar_clave("X", "x" * 30)
        self.assertTrue(ok)


class TestMain(unittest.TestCase):
    def test_main_no_falla_sin_claves(self):
        for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HUGGINGFACE_TOKEN"):
            os.environ.pop(k, None)
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertIn(rc, (0, 1))
        salida = buffer.getvalue()
        import json
        cargado = json.loads(salida)
        self.assertIn("OPENAI_API_KEY", cargado)


if __name__ == "__main__":
    unittest.main()
