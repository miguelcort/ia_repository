"""
Lección: 21-teoria-de-grafos
Fase: 01
Prerrequisitos: 12-operaciones-con-tensores
"""
from __future__ import annotations
import sys
from collections import deque
import numpy as np


def bfs(grafo, inicio):
    """Breadth-first search. Devuelve orden de visita."""
    visitados = {inicio}
    cola = deque([inicio])
    orden = []
    while cola:
        nodo = cola.popleft()
        orden.append(nodo)
        for vecino in grafo.get(nodo, []):
            if vecino not in visitados:
                visitados.add(vecino)
                cola.append(vecino)
    return orden


def dfs(grafo, inicio):
    """Depth-first search."""
    visitados = set()
    orden = []
    def _dfs(nodo):
        visitados.add(nodo)
        orden.append(nodo)
        for vecino in grafo.get(nodo, []):
            if vecino not in visitados:
                _dfs(vecino)
    _dfs(inicio)
    return orden


def matriz_adyacencia(grafo, nodos):
    """Construye matriz de adyacencia N x N."""
    idx = {n: i for i, n in enumerate(nodos)}
    A = np.zeros((len(nodos), len(nodos)), dtype=int)
    for u, vecinos in grafo.items():
        for v in vecinos:
            A[idx[u], idx[v]] = 1
    return A


def main() -> int:
    grafo = {
        "A": ["B", "C"],
        "B": ["D"],
        "C": ["D"],
        "D": [],
    }
    print(f"BFS desde A: {bfs(grafo, 'A')}")
    print(f"DFS desde A: {dfs(grafo, 'A')}")
    print(f"Matriz adyacencia:\n{matriz_adyacencia(grafo, list(grafo))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())