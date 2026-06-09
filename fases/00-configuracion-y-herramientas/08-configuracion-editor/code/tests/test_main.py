"""Pruebas para el verificador de configuración de VS Code."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402

sys.path[:] = RAIZ


class TestValidarSettings(unittest.TestCase):
    def test_settings_completo_no_reporta_problemas(self):
        datos = {
            "python.defaultInterpreterPath": ".venv/bin/python",
            "editor.formatOnSave": True,
            "[python]": {"editor.defaultFormatter": "ms-python.black-formatter"},
        }
        self.assertEqual(main.validar_settings(datos), [])

    def test_falta_interpreter(self):
        datos = {
            "editor.formatOnSave": True,
            "[python]": {"editor.defaultFormatter": "ms-python.black-formatter"},
        }
        problemas = main.validar_settings(datos)
        self.assertTrue(any("defaultInterpreterPath" in p for p in problemas))

    def test_format_on_save_deshabilitado(self):
        datos = {
            "python.defaultInterpreterPath": ".venv/bin/python",
            "editor.formatOnSave": False,
            "[python]": {"editor.defaultFormatter": "ms-python.black-formatter"},
        }
        problemas = main.validar_settings(datos)
        self.assertTrue(any("formatOnSave" in p for p in problemas))

    def test_falta_formatter_python(self):
        datos = {
            "python.defaultInterpreterPath": ".venv/bin/python",
            "editor.formatOnSave": True,
        }
        problemas = main.validar_settings(datos)
        self.assertTrue(any("defaultFormatter" in p for p in problemas))


class TestValidarExtensions(unittest.TestCase):
    def test_tiene_las_dos_obligatorias(self):
        datos = {
            "recommendations": [
                "ms-python.python",
                "ms-python.vscode-pylance",
            ]
        }
        self.assertEqual(main.validar_extensions(datos), [])

    def test_falta_python(self):
        datos = {"recommendations": ["ms-python.vscode-pylance"]}
        problemas = main.validar_extensions(datos)
        self.assertTrue(any("ms-python.python" in p for p in problemas))

    def test_falta_pylance(self):
        datos = {"recommendations": ["ms-python.python"]}
        problemas = main.validar_extensions(datos)
        self.assertTrue(any("pylance" in p for p in problemas))

    def test_sin_recommendations_es_invalido(self):
        problemas = main.validar_extensions({})
        self.assertGreater(len(problemas), 0)


class TestCargarJson(unittest.TestCase):
    def test_carga_archivo_valido(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"a": 1}, f)
            ruta = Path(f.name)
        try:
            datos = main.cargar_json(ruta)
            self.assertEqual(datos, {"a": 1})
        finally:
            ruta.unlink()


class TestVerificarDirectorio(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_sin_settings_ni_extensions(self):
        reporte = main.verificar_directorio(Path(self.tmp))
        self.assertFalse(reporte["settings_ok"])
        self.assertFalse(reporte["extensions_ok"])
        self.assertGreater(len(reporte["problemas"]), 0)

    def test_settings_y_extensions_completos(self):
        (Path(self.tmp) / "settings.json").write_text(json.dumps({
            "python.defaultInterpreterPath": ".venv/bin/python",
            "editor.formatOnSave": True,
            "[python]": {"editor.defaultFormatter": "ms-python.black-formatter"},
        }))
        (Path(self.tmp) / "extensions.json").write_text(json.dumps({
            "recommendations": ["ms-python.python", "ms-python.vscode-pylance"]
        }))
        reporte = main.verificar_directorio(Path(self.tmp))
        self.assertTrue(reporte["settings_ok"])
        self.assertTrue(reporte["extensions_ok"])
        self.assertEqual(reporte["problemas"], [])

    def test_json_invalido(self):
        (Path(self.tmp) / "settings.json").write_text("{esto no es json")
        reporte = main.verificar_directorio(Path(self.tmp))
        self.assertFalse(reporte["settings_ok"])
        self.assertTrue(any("JSON invalido" in p for p in reporte["problemas"]))


class TestMain(unittest.TestCase):
    def test_main_sin_vscode(self):
        import io
        from contextlib import redirect_stdout
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                rc = main.main()
            self.assertEqual(rc, 1)
            self.assertIn("AVISO", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
