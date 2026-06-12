"""
Lección: 19-swarm-optimization-pso-aco
Fase: 16
Swarm optimization: Particle Swarm
Optimization, Ant Colony Optimization,
swarm intelligence for combinatorial
optimization.
"""
from __future__ import annotations
import math
import random


class Particle:
    def __init__(self, dim, bounds):
        self.position = [random.uniform(b[0], b[1]) for b in bounds]
        self.velocity = [random.uniform(-1, 1) for _ in range(dim)]
        self.best_position = list(self.position)
        self.best_value = float("inf")


def pso(objective, bounds, n_particles=10, n_iters=20, w=0.5, c1=1.5, c2=1.5, seed=None):
    if seed is not None:
        random.seed(seed)
    dim = len(bounds)
    particles = [Particle(dim, bounds) for _ in range(n_particles)]
    global_best_position = None
    global_best_value = float("inf")
    for it in range(n_iters):
        for p in particles:
            value = objective(p.position)
            if value < p.best_value:
                p.best_value = value
                p.best_position = list(p.position)
            if value < global_best_value:
                global_best_value = value
                global_best_position = list(p.position)
        for p in particles:
            for d in range(dim):
                r1 = random.random()
                r2 = random.random()
                p.velocity[d] = (
                    w * p.velocity[d]
                    + c1 * r1 * (p.best_position[d] - p.position[d])
                    + c2 * r2 * (global_best_position[d] - p.position[d])
                )
                p.position[d] += p.velocity[d]
    return global_best_position, global_best_value


def aco_tsp(distances, n_ants=5, n_iters=20, alpha=1.0, beta=2.0, evaporation=0.5, seed=None):
    if seed is not None:
        random.seed(seed)
    n = len(distances)
    pheromone = [[1.0] * n for _ in range(n)]
    best_path = None
    best_length = float("inf")
    for it in range(n_iters):
        paths = []
        for ant in range(n_ants):
            visited = [0]
            unvisited = list(range(1, n))
            current = 0
            while unvisited:
                probs = []
                for j in unvisited:
                    tau = pheromone[current][j] ** alpha
                    eta = (1.0 / distances[current][j]) ** beta if distances[current][j] > 0 else 0
                    probs.append(tau * eta)
                total = sum(probs)
                probs = [p / total for p in probs]
                r = random.random()
                cum = 0
                chosen = unvisited[0]
                for j, p in zip(unvisited, probs):
                    cum += p
                    if r < cum:
                        chosen = j
                        break
                visited.append(chosen)
                unvisited.remove(chosen)
                current = chosen
            length = sum(distances[visited[i]][visited[(i+1) % n]] for i in range(n))
            paths.append((visited, length))
            if length < best_length:
                best_length = length
                best_path = list(visited)
        for i in range(n):
            for j in range(n):
                pheromone[i][j] *= (1 - evaporation)
        for path, length in paths:
            for i in range(n):
                a = path[i]
                b = path[(i+1) % n]
                pheromone[a][b] += 1.0 / length
                pheromone[b][a] += 1.0 / length
    return best_path, best_length


def main() -> int:
    def sphere(x):
        return sum(xi ** 2 for xi in x)
    pos, val = pso(sphere, [(-5, 5), (-5, 5)], n_particles=10, n_iters=20, seed=42)
    print(f"PSO: {val:.4f}")
    distances = [[0, 1, 2, 3], [1, 0, 4, 1], [2, 4, 0, 2], [3, 1, 2, 0]]
    path, length = aco_tsp(distances, n_ants=5, n_iters=10, seed=42)
    print(f"ACO: {length}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())