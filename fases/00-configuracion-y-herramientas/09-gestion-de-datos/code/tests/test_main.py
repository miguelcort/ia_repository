"""Pruebas para 09-gestion-de-datos."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402

sys.path[:] = RAIZ


def escribir_csv(contenido: str) -> Path:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
    )
    f.write(contenido)
    f.close()
    return Path(f.name)


class TestSha256(unittest.TestCase):
    def test_mismo_contenido_mismo_hash(self):
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".bin", delete=False
        ) as f:
            f.write(b"hola mundo")
            ruta = Path(f.name)
        try:
            h1 = main.sha256_archivo(ruta)
            h2 = main.sha256_archivo(ruta)
            self.assertEqual(h1, h2)
            self.assertEqual(len(h1), 64)
        finally:
            ruta.unlink()

    def test_contenido_diferente_hash_diferente(self):
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".bin", delete=False
        ) as f1, tempfile.NamedTemporaryFile(
            mode="wb", suffix=".bin", delete=False
        ) as f2:
            f1.write(b"a")
            f2.write(b"b")
            r1, r2 = Path(f1.name), Path(f2.name)
        try:
            self.assertNotEqual(main.sha256_archivo(r1), main.sha256_archivo(r2))
        finally:
            r1.unlink()
            r2.unlink()


class TestCargarCsv(unittest.TestCase):
    def test_carga_basica(self):
        ruta = escribir_csv("a,b,c\n1,2,3\n4,5,6\n")
        try:
            cabecera, filas = main.cargar_csv(ruta)
            self.assertEqual(cabecera, ["a", "b", "c"])
            self.assertEqual(filas, [["1", "2", "3"], ["4", "5", "6"]])
        finally:
            ruta.unlink()

    def test_csv_vacio(self):
        ruta = escribir_csv("")
        try:
            cabecera, filas = main.cargar_csv(ruta)
            self.assertEqual(cabecera, [])
            self.assertEqual(filas, [])
        finally:
            ruta.unlink()

    def test_csv_solo_cabecera(self):
        ruta = escribir_csv("a,b\n")
        try:
            cabecera, filas = main.cargar_csv(ruta)
            self.assertEqual(cabecera, ["a", "b"])
            self.assertEqual(filas, [])
        finally:
            ruta.unlink()


class TestValidarNoVacio(unittest.TestCase):
    def test_sin_cabecera_reporta_problema(self):
        self.assertIn("CSV sin cabecera", main.validar_no_vacio([], [["x"]]))

    def test_sin_filas_reporta_problema(self):
        self.assertIn("CSV sin filas", main.validar_no_vacio(["a"], []))

    def test_columnas_inconsistentes(self):
        problemas = main.validar_no_vacio(["a", "b"], [["1", "2"], ["3"]])
        self.assertTrue(any("fila 1" in p for p in problemas))

    def test_columnas_consistentes(self):
        self.assertEqual(main.validar_no_vacio(["a", "b"], [["1", "2"]]), [])


class TestGenerarManifiesto(unittest.TestCase):
    def test_manifiesto_completo(self):
        ruta = escribir_csv("a,b\n1,2\n3,4\n5,6\n")
        try:
            m = main.generar_manifiesto(ruta)
            self.assertEqual(m["filas"], 3)
            self.assertEqual(m["columnas"], ["a", "b"])
            self.assertEqual(m["tamano_bytes"], ruta.stat().st_size)
            self.assertEqual(len(m["hash_sha256"]), 64)
        finally:
            ruta.unlink()


class TestMain(unittest.TestCase):
    def test_sin_argumentos(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main([])
        self.assertEqual(rc, 1)
        self.assertIn("Uso:", buffer.getvalue())

    def test_archivo_inexistente(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main(["/no/existe.csv"])
        self.assertEqual(rc, 1)

    def test_csv_valido(self):
        import io
        from contextlib import redirect_stdout
        import json
        ruta = escribir_csv("a,b\n1,2\n3,4\n")
        try:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                rc = main.main([str(ruta)])
            self.assertEqual(rc, 0)
            cargado = json.loads(buffer.getvalue())
            self.assertEqual(cargado["filas"], 2)
        finally:
            ruta.unlink()


if __name__ == "__main__":
    unittest.main()
