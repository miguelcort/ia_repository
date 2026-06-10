"""Pruebas para 03-cnns-desde-lenet-hasta-resnet."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestBloque(unittest.TestCase):
    def test_conv_bn_relu_mismo_tamano(self):
        out = main.bloque_conv_bn_relu((28, 28, 1), 1, 16, k=3)
        self.assertEqual(out, (28, 28, 16))

    def test_residual_mismos_canales(self):
        out = main.bloque_residual((28, 28, 16), 16)
        self.assertEqual(out, (28, 28, 16))

    def test_residual_proyeccion(self):
        out = main.bloque_residual((28, 28, 16), 32)
        self.assertEqual(out, (28, 28, 32))


class TestReceptiveField(unittest.TestCase):
    def test_una_capa_conv3(self):
        # RF = 1 + 2 = 3
        self.assertEqual(main.receptive_field(["conv3"]), 3)

    def test_pool_y_conv(self):
        # pool2: stride=2, conv3: RF += 2*2 = 4, total RF = 1 + 4 = 5
        self.assertEqual(main.receptive_field(["pool2", "conv3"]), 5)


class TestParams(unittest.TestCase):
    def test_conv_3x3(self):
        # 3 canales in, 16 out, k=3: 3*9*16 + 16 = 432 + 16 = 448
        self.assertEqual(main.cuente_params_conv(3, 16, 3), 448)

    def test_fc(self):
        # 100 in, 10 out: 100*10 + 10 = 1010
        self.assertEqual(main.cuente_params_fc(100, 10), 1010)


class TestLeNet(unittest.TestCase):
    def test_total(self):
        p = main.total_params_lenet()
        self.assertGreater(p["total"], 50000)
        self.assertLess(p["total"], 100000)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("LeNet", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()