# Teoria de grafos para ML

> GNNs son el equivalente de CNNs para datos no estructurados como moleculas o redes sociales.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12-operaciones-con-tensores
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar BFS y DFS.
- Construir matriz de adyacencia.
- Diagnosticar el algoritmo de grafos adecuado.

## Constrúyelo

```python
from collections import deque


def bfs(grafo, inicio):
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
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-gnn-elegir
fase: 01
leccion: 21
---

1. <10K nodos: GCN.
2. Grandes: GraphSAGE o ClusterGCN.
3. Heterogeneos: RGCN.
4. Link prediction: Node2Vec.
5. Pooling: GNN jerarquico.
```

## Ejercicios

1. **Dijkstra**: implementa shortest path en grafo ponderado.
2. **PageRank**: implementa el algoritmo original.
3. **Desafio**: implementa una GCN simple con NumPy.

## Lecturas recomendadas

- PyG (PyTorch Geometric): <https://pytorch-geometric.readthedocs.io/>
- DGL: <https://www.dgl.ai/>
- "Graph Representation Learning" (Hamilton)

---

> 📚 **Adaptación al español** de la lección "[Graph Theory for ML]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).