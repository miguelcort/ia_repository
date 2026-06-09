"""Pruebas para 07-docker-para-ia.

Como el Dockerfile no se puede construir sin Docker instalado,
las pruebas validan la sintaxis y la convencion del archivo.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402

sys.path[:] = RAIZ

DOCKERFILE = Path(__file__).resolve().parent.parent / "Dockerfile"


class TestDockerfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contenido = DOCKERFILE.read_text(encoding="utf-8")

    def test_existe(self):
        self.assertTrue(DOCKERFILE.exists(), "Falta el Dockerfile")

    def test_python_3_11_o_superior(self):
        m = re.search(r"FROM python:(\d+)\.(\d+)", self.contenido)
        self.assertIsNotNone(m, "FROM python:X.Y no encontrado")
        self.assertGreaterEqual((int(m.group(1)), int(m.group(2))), (3, 11))

    def test_multi_stage(self):
        self.assertIn(" AS builder", self.contenido, "Falta stage builder")
        self.assertIn(" AS runtime", self.contenido, "Falta stage runtime")

    def test_workdir_explicito(self):
        self.assertRegex(self.contenido, r"WORKDIR\s+/app", "WORKDIR /app no encontrado")

    def test_usuario_no_root(self):
        self.assertRegex(self.contenido, r"USER\s+app\b", "USER app no encontrado")

    def test_cmd_en_forma_exec(self):
        m = re.search(r'CMD\s+(\S.*)', self.contenido)
        self.assertIsNotNone(m)
        cmd = m.group(1).strip()
        self.assertTrue(
            cmd.startswith("["),
            f"CMD debe estar en forma exec (lista), no shell. Actual: {cmd!r}",
        )

    def test_requirements_copiado_antes_de_codigo(self):
        pos_req = self.contenido.find("requirements.txt")
        pos_code = self.contenido.find("COPY --chown=")
        self.assertGreater(pos_req, 0)
        self.assertGreater(pos_code, 0)
        self.assertLess(pos_req, pos_code, "requirements.txt debe copiarse antes que el código")


class TestMain(unittest.TestCase):
    def test_main_sin_args_imprime_demo(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main([])
        self.assertEqual(rc, 0)
        self.assertIn("Leccion 07", buffer.getvalue())

    def test_version_imprime_json(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main(["--version"])
        self.assertEqual(rc, 0)
        import json
        data = json.loads(buffer.getvalue())
        self.assertEqual(data["app"], "leccion-07-docker-para-ia")
        self.assertIn("python", data)
        self.assertIn("sistema", data)
        self.assertIn("docker", data)


if __name__ == "__main__":
    unittest.main()
