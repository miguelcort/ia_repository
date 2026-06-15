# 21 — Teoría de grafos para ML

> Los transformers, las redes neuronales de grafos, los sistemas de recomendación y el routing de agentes son grafos en el fondo.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-vectores-matrices-operaciones
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Representar grafos con matrices de adyacencia y listas de
  adyacencia.
- Implementar BFS, DFS y componentes conexas desde cero.
- Aplicar PageRank y random walks sobre grafos.
- Diagnosticar cuándo una tarea de ML se modela mejor como
  grafo.

## El problema

Si tus datos son "cosas y relaciones entre cosas", un grafo es
la representación natural. Recomendaciones (usuarios y
productos), moléculas (átomos y enlaces), redes sociales
(personas y amistades), grafos de conocimiento (entidades y
relaciones), agentes (nodos de decisión y llamadas a tools):
todos son grafos. La lección cubre el mínimo de teoría de
grafos para que el estudiante pueda leer papers de GNN,
implementar PageRank, y razonar sobre sistemas basados en
grafos.

## El concepto

**Definiciones.** Un grafo `G = (V, E)` tiene vértices `V` y
aristas `E`. Puede ser **dirigido** o **no dirigido**, **ponderado**
o **no ponderado**. Un **multigrafo** permite aristas múltiples
entre los mismos nodos. Un **grafo bipartito** tiene dos
conjuntos de nodos con aristas solo entre conjuntos (películas
y usuarios, por ejemplo).

**Representaciones.**

- **Lista de aristas**: `[(0,1), (1,2), (2,0)]`. Compacta para
  grafos dispersos.
- **Matriz de adyacencia**: `A[i][j] = 1` si hay arista
  `i → j`. `O(V²)` espacio, `O(1)` lookup, ideal para grafos
  densos.
- **Lista de adyacencia**: `dict[int, list[int]]`. `O(V + E)`
  espacio, `O(grado)` lookup, ideal para grafos dispersos.

**BFS (Breadth-First Search).** Recorre por niveles, encuentra
el camino más corto en grafos no ponderados. `O(V + E)`.

**DFS (Depth-First Search).** Recorre por profundidad, útil
para detectar ciclos, ordenar topológicamente, encontrar
componentes conexas. `O(V + E)`.

**PageRank.** El algoritmo que hizo a Google. Asigna a cada
página una puntuación proporcional a la suma de las
puntuaciones de las páginas que la enlazan, dividido por su
número de enlaces salientes. Es un eigenvector del *Google
matrix* `(1-d)/N + d · M` donde `M` es la versión normalizada
de la matriz de adyacencia y `d` es el *damping factor* (típico
0.85). Convergencia en `O(iteraciones · E)`.

**Random walks.** Un caminante aleatorio sobre el grafo: en
cada paso, salta a un vecino uniformemente al azar. La
distribución estacionaria del caminante (si existe) es la
*PageRank* del grafo. Los random walks son la base de *node2vec*,
DeepWalk y muchos *graph embeddings*.

**GNN (Graph Neural Networks).** Generalización de CNN a grafos.
Cada nodo agrega los *features* de sus vecinos, transformados
por una red neuronal. Después de K rondas, cada nodo tiene
información de su K-hop neighborhood. *GraphSAGE*, *GAT* y
*GCN* son los tres modelos canónicos.

**Aplicaciones en IA moderna.**

| Aplicación | Cómo usa grafos |
|---|---|
| **GNN** | Predicción de nodos, aristas, grafos completos |
| **GraphRAG** | RAG sobre grafo de conocimiento |
| **PageRank** | Ranking de documentos, recomendaciones |
| **Atención de transformers** | Grafo completo auto-atendido |
| **Routing de agentes** | Grafo de tools y decisiones |

## Constrúyelo

```python
from collections import defaultdict, deque
import numpy as np


class Grafo:
    """Grafo dirigido como lista de adyacencia."""

    def __init__(self):
        self.adj: dict[int, list[int]] = defaultdict(list)

    def agregar_arista(self, u, v):
        self.adj[u].append(v)

    def bfs(self, inicio):
        """BFS: devuelve orden de visita desde inicio."""
        visitados, cola = {inicio}, deque([inicio])
        orden = []
        while cola:
            u = cola.popleft()
            orden.append(u)
            for v in self.adj[u]:
                if v not in visitados:
                    visitados.add(v)
                    cola.append(v)
        return orden

    def dfs(self, inicio):
        """DFS: devuelve orden de visita desde inicio."""
        visitados, orden = set(), []

        def _dfs(u):
            visitados.add(u)
            orden.append(u)
            for v in self.adj[u]:
                if v not in visitados:
                    _dfs(v)

        _dfs(inicio)
        return orden

    def componentes_conexas(self):
        """DFS desde cada nodo no visitado, agrupando visitas."""
        visitados, comps = set(), []
        for u in list(self.adj.keys()):
            if u not in visitados:
                comp = []
                stack = [u]
                while stack:
                    n = stack.pop()
                    if n in visitados:
                        continue
                    visitados.add(n)
                    comp.append(n)
                    stack.extend(self.adj[n])
                comps.append(comp)
        return comps

    def pagerank(self, d=0.85, n_iter=100, tol=1e-6):
        """PageRank con damping factor d."""
        nodos = list(self.adj.keys())
        N = len(nodos)
        idx = {n: i for i, n in enumerate(nodos)}
        # matriz de transición estocástica
        M = np.zeros((N, N))
        for u, vecinos in self.adj.items():
            if not vecinos:
                M[idx[u], :] = 1.0 / N  # sumidero: teletransporte uniforme
            else:
                for v in vecinos:
                    M[idx[v], idx[u]] = 1.0 / len(vecinos)
        v = np.ones(N) / N
        for _ in range(n_iter):
            v_new = (1 - d) / N + d * (M @ v)
            if np.linalg.norm(v_new - v, 1) < tol:
                break
            v = v_new
        return {nodos[i]: float(v[i]) for i in range(N)}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Ejercicios

1. **PageRank**: implementa PageRank sobre un grafo de
   ejemplo (4-5 páginas) y verifica que las páginas con más
   enlaces entrantes obtienen mayor ranking.
2. **Random walk**: implementa un random walk sobre un grafo
   bipartito y verifica la distribución estacionaria.
3. **Desafío**: implementa Louvain para detección de
   comunidades (complejidad `O(n log n)`).

## Lecturas recomendadas

- *Network Science* — Barabási.
- *Graph Neural Networks: A Review of Methods and Applications*
  — Zhou et al.
- PyTorch Geometric: <https://pytorch-geometric.readthedocs.io>.
- DGL: <https://www.dgl.ai>.

---

> 📚 **Adaptación al español** de la lección "[Graph Theory for ML]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
