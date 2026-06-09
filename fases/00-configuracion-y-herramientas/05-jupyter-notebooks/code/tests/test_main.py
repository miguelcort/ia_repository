"""Pruebas para el verificador de notebooks."""
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


def notebook_ejemplo(bien_formado: bool = True) -> dict:
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["import numpy as np\n", "import pandas as pd\n"],
                "outputs": [],
                "execution_count": None,
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["np.random.seed(42)\n", "datos = pd.read_csv('x.csv')\n"],
                "outputs": [],
                "execution_count": None,
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Analisis\n", "Este es el analisis del experimento."],
            },
        ],
        "metadata": {"kernelspec": {"name": "python3"}},
        "nbformat": 4,
    }
    if not bien_formado:
        nb["cells"][0]["source"] = ["# sin imports\n"]
    return nb


class TestCargarNotebook(unittest.TestCase):
    def test_carga_desde_archivo(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ipynb", delete=False, encoding="utf-8"
        ) as f:
            json.dump(notebook_ejemplo(), f)
            ruta = Path(f.name)
        try:
            nb = main.cargar_notebook(ruta)
            self.assertIn("cells", nb)
        finally:
            ruta.unlink()


class TestObtenerCodigo(unittest.TestCase):
    def test_concatena_source(self):
        celda = {"source": ["a = 1\n", "b = 2\n"]}
        self.assertEqual(main.obtener_codigo(celda), "a = 1\nb = 2\n")

    def test_con_lista_vacia(self):
        self.assertEqual(main.obtener_codigo({"source": []}), "")


class TestValidarEstructura(unittest.TestCase):
    def test_notebook_vacio(self):
        problemas = main.validar_estructura({"cells": []})
        self.assertIn("notebook vacio", problemas)

    def test_primera_celda_sin_imports(self):
        nb = notebook_ejemplo(bien_formado=False)
        problemas = main.validar_estructura(nb)
        self.assertTrue(any("imports" in p for p in problemas))

    def test_sin_celdas_de_codigo(self):
        nb = {
            "cells": [
                {"cell_type": "markdown", "source": ["# titulo"]},
            ]
        }
        problemas = main.validar_estructura(nb)
        self.assertIn("sin celdas de codigo", problemas)

    def test_usa_random_sin_semilla(self):
        nb = {
            "cells": [
                {"cell_type": "code", "source": ["import random\n", "x = random.random()"]},
            ]
        }
        problemas = main.validar_estructura(nb)
        self.assertTrue(any("semilla" in p for p in problemas))

    def test_notebook_bien_formado(self):
        nb = notebook_ejemplo(bien_formado=True)
        problemas = main.validar_estructura(nb)
        self.assertEqual(problemas, [])


class TestContarPalabrasMarkdown(unittest.TestCase):
    def test_cuenta_palabras(self):
        nb = {
            "cells": [
                {"cell_type": "markdown", "source": ["uno dos tres cuatro"]},
                {"cell_type": "code", "source": ["x = 1"]},
                {"cell_type": "markdown", "source": ["cinco seis"]},
            ]
        }
        self.assertEqual(main.contar_palabras_markdown(nb), 6)


class TestMain(unittest.TestCase):
    def test_main_sin_argumentos(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main(argv=[])
        self.assertEqual(rc, 1)
        self.assertIn("Uso:", buffer.getvalue())

    def test_main_archivo_inexistente(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main(argv=["/no/existe.ipynb"])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
