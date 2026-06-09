"""Pruebas para 21-teoria-de-grafos."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestBFS(unittest.TestCase):
    def test_bfs_grafo_simple(self):
        g = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}
        visitados = main.bfs(g, "A")
        self.assertEqual(visitados[0], "A")
        self.assertEqual(set(visitados), {"A", "B", "C", "D"})

    def test_bfs_nodo_aislado(self):
        g = {"A": [], "B": []}
        self.assertEqual(main.bfs(g, "A"), ["A"])


class TestDFS(unittest.TestCase):
    def test_dfs(self):
        g = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}
        visitados = main.dfs(g, "A")
        self.assertEqual(visitados[0], "A")
        self.assertEqual(set(visitados), {"A", "B", "C", "D"})


class TestMatriz(unittest.TestCase):
    def test_matriz_adyacencia(self):
        g = {"A": ["B"], "B": ["A"], "C": []}
        nodos = ["A", "B", "C"]
        A = main.matriz_adyacencia(g, nodos)
        # A->B y B->A
        self.assertEqual(A[0, 1], 1)
        self.assertEqual(A[1, 0], 1)
        # C es aislado
        self.assertEqual(A[0, 2], 0)
        self.assertEqual(A[2, 0], 0)

    def test_matriz_simetrica_para_no_dirigido(self):
        g = {"A": ["B"], "B": ["A"]}
        A = main.matriz_adyacencia(g, ["A", "B"])
        np.testing.assert_array_equal(A, A.T)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("BFS", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()