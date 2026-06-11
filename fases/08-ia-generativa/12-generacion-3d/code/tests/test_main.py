"""Pruebas para 12-generacion-3d."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestRays(unittest.TestCase):
    def test_rays_shape(self):
        camera = np.array([0.0, 0.0, 0.0])
        c2w = np.eye(4)
        rays_o, rays_d = main.rays_o_d(camera, (4, 8), focal=1.0, c2w=c2w)
        self.assertEqual(rays_o.shape, (4, 8, 3))
        self.assertEqual(rays_d.shape, (4, 8, 3))


class TestVolumetric(unittest.TestCase):
    def test_shape(self):
        rgb = np.random.default_rng(0).standard_normal((2, 16, 3))
        sigma = np.random.default_rng(1).standard_normal((2, 16, 1))
        t_vals = np.linspace(0, 1, 16)
        out = main.volumetric_render(rgb, sigma, t_vals)
        self.assertEqual(out.shape, (2, 3))


class TestGaussians(unittest.TestCase):
    def test_init(self):
        gauss = main.gaussians_init(50, seed=0)
        self.assertEqual(gauss["positions"].shape, (50, 3))
        self.assertEqual(gauss["colors"].shape, (50, 3))
        # Rotation identidad
        np.testing.assert_array_equal(gauss["rotations"][0], [1, 0, 0, 0])


class TestProjection(unittest.TestCase):
    def test_shape(self):
        positions = np.random.default_rng(0).standard_normal((10, 3))
        K = np.eye(3)
        c2w = np.eye(4)
        uv, depth = main.projection_3d_to_2d(positions, K, c2w)
        self.assertEqual(uv.shape, (10, 2))
        self.assertEqual(depth.shape, (10,))


class TestComponents(unittest.TestCase):
    def test_nerf_seis(self):
        comps = main.nerf_components()
        self.assertEqual(len(comps), 6)

    def test_gs_cinco(self):
        comps = main.gs_components()
        self.assertEqual(len(comps), 5)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()