"""Pruebas para el verificador de entorno."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402

sys.path[:] = RAIZ


class TestCheck(unittest.TestCase):
    def test_check_pasa_con_predicado_verdadero(self):
        resultado = main.check("ejemplo", lambda: True, lambda: "ok")
        self.assertEqual(resultado, ("ejemplo", "pass", "ok"))

    def test_check_falla_con_predicado_falso(self):
        resultado = main.check("ejemplo", lambda: False)
        self.assertEqual(resultado[0], "ejemplo")
        self.assertEqual(resultado[1], "fail")
        self.assertIsNone(resultado[2])

    def test_check_captura_excepcion_como_fail(self):
        def explotar():
            raise RuntimeError("boom")
        resultado = main.check("ejemplo", explotar)
        self.assertEqual(resultado[1], "fail")
        self.assertIn("boom", resultado[2])

    def test_check_sin_detalle_no_lo_llama(self):
        llamado = []

        def detail():
            llamado.append(True)
            return "valor"

        # Cuando se pasa detail, sí debe llamarse si el check pasa.
        main.check("ok", lambda: True, detail)
        self.assertEqual(llamado, [True])

    def test_check_sin_detalle_es_none(self):
        resultado = main.check("ok", lambda: True)
        self.assertIsNone(resultado[2])


class TestCheckPython(unittest.TestCase):
    def test_python_actual_pasa_si_es_3_10_o_superior(self):
        import sys as _sys
        mayor, menor = _sys.version_info[:2]
        nombre, estado, detalle = main.check_python()
        self.assertEqual(nombre, "python")
        if (mayor, menor) >= (3, 10):
            self.assertEqual(estado, "pass")
            self.assertRegex(detalle, r"^\d+\.\d+\.\d+$")
        else:
            self.assertEqual(estado, "fail")
            self.assertIn(str(mayor), detalle)
            self.assertIn(str(menor), detalle)


class TestCheckImport(unittest.TestCase):
    def test_modulo_existente(self):
        nombre, estado, _ = main.check_import("sys")
        self.assertEqual((nombre, estado), ("sys", "pass"))

    def test_modulo_inexistente_falla(self):
        nombre, estado, detalle = main.check_import("modulo_que_no_existe_123")
        self.assertEqual((nombre, estado), ("modulo_que_no_existe_123", "fail"))
        self.assertIsNotNone(detalle)


class TestCheckCommand(unittest.TestCase):
    def test_python_existe_en_path(self):
        nombre, estado, _ = main.check_command("python3")
        self.assertEqual(nombre, "python3")
        self.assertEqual(estado, "pass")

    def test_comando_inexistente_falla(self):
        nombre, estado, _ = main.check_command("comando_inexistente_xyz_123")
        self.assertEqual((nombre, estado), ("comando_inexistente_xyz_123", "fail"))


class TestVerificarEntorno(unittest.TestCase):
    def test_estructura_del_reporte(self):
        reporte = main.verificar_entorno()
        self.assertIn("sistema", reporte)
        self.assertIn("python", reporte)
        self.assertIn("checks", reporte)
        self.assertIsInstance(reporte["checks"], list)
        self.assertGreater(len(reporte["checks"]), 0)
        for c in reporte["checks"]:
            self.assertIn("nombre", c)
            self.assertIn("estado", c)
            self.assertIn("detalle", c)
            self.assertIn(c["estado"], {"pass", "warn", "fail"})

    def test_reporte_es_json_serializable(self):
        reporte = main.verificar_entorno()
        texto = json.dumps(reporte, ensure_ascii=False)
        cargado = json.loads(texto)
        self.assertEqual(cargado["python"], reporte["python"])


class TestMain(unittest.TestCase):
    def test_main_retorna_int(self):
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertIn(rc, (0, 1))
        salida = buffer.getvalue()
        self.assertIn("checks", salida)


if __name__ == "__main__":
    unittest.main()
