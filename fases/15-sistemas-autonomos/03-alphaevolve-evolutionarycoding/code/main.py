"""
Lección: 03-alphaevolve-evolutionarycoding
Fase: 15
AlphaEvolve (Google DeepMind 2025): evolutionary coding
agent. Code mutation + selection on benchmarks. +SOTA on
math + code.
"""
from __future__ import annotations
import random


def mutate_code(code, n_mutations=1):
    """Mock code mutation: replace, insert, delete chars."""
    if not code:
        return code
    code = list(code)
    for _ in range(n_mutations):
        if not code:
            break
        op = random.choice(["replace", "insert", "delete"])
        if op == "replace" and code:
            i = random.randint(0, len(code) - 1)
            code[i] = random.choice("abcdefghijklmnopqrstuvwxyz ")
        elif op == "insert":
            i = random.randint(0, len(code))
            code.insert(i, random.choice("abcdefghijklmnopqrstuvwxyz"))
        elif op == "delete" and code:
            i = random.randint(0, len(code) - 1)
            del code[i]
    return "".join(code)


def evaluate_code(code, eval_fn):
    """Evaluate code with benchmark."""
    return eval_fn(code)


def alphaevolve(initial_code, eval_fn, n_generations=50, population_size=20):
    """Evolutionary coding agent: mutate + select."""
    population = [initial_code] + [mutate_code(initial_code, n_mutations=2) for _ in range(population_size - 1)]
    for gen in range(n_generations):
        scored = [(c, eval_fn(c)) for c in population]
        scored.sort(key=lambda x: x[1], reverse=True)
        # keep top half
        top = [c for c, s in scored[:population_size // 2]]
        # mutate top to form new generation
        new_pop = list(top)
        while len(new_pop) < population_size:
            parent = random.choice(top)
            new_pop.append(mutate_code(parent, n_mutations=1))
        population = new_pop
    # best from last population
    scored = [(c, eval_fn(c)) for c in population]
    return max(scored, key=lambda x: x[1])


def main() -> int:
    # simple benchmark: maximize sum of ord(c) for c in code
    def eval_fn(code):
        return sum(ord(c) for c in code) if code else 0
    initial = "abc"
    best_code, best_score = alphaevolve(initial, eval_fn, n_generations=20, population_size=10)
    print(f"Best: {best_score}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())