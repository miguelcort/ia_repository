"""
Lección: 11-planning-htn-and-evolutionary
Fase: 14
Planning: HTN (Hierarchical Task Network) y evolutionary.
ADaPT, SayCan, SayPlan, RAP. Hierarchical + evolutionary
+ LLM-based planning.
"""
from __future__ import annotations
import random


def htn_decompose(task, methods):
    """Decompose task into subtasks using HTN methods."""
    if task not in methods:
        return [task]  # primitive
    method = methods[task]
    return method["subtasks"]


def htn_plan(task, methods, max_depth=10):
    """HTN planning: recursively decompose."""
    if max_depth <= 0:
        return [task]
    subtasks = htn_decompose(task, methods)
    plan = []
    for sub in subtasks:
        if sub in methods:
            plan.extend(htn_plan(sub, methods, max_depth - 1))
        else:
            plan.append(sub)
    return plan


def evolutionary_plan(initial_plan, evaluate_fn, mutate_fn, n_generations=50, population_size=10):
    """Evolutionary planning: evolve plan via mutation + selection."""
    population = [list(initial_plan) for _ in range(population_size)]
    # random mutations
    for _ in range(population_size - 1):
        p = list(initial_plan)
        if random.random() < 0.5 and len(p) > 1:
            i, j = random.sample(range(len(p)), 2)
            p[i], p[j] = p[j], p[i]
        population.append(p)
    for gen in range(n_generations):
        scored = [(p, evaluate_fn(p)) for p in population]
        scored.sort(key=lambda x: x[1], reverse=True)
        # keep top half
        population = [p for p, s in scored[:population_size]]
        # mutate
        for _ in range(population_size):
            p = mutate_fn(population[0])
            population.append(p)
    return population[0]


def llm_plan(query, plan_fn):
    """LLM-based planning (RAP-style)."""
    return plan_fn(query)


def main() -> int:
    # HTN example: cook_dinner -> [chop, cook, plate]
    methods = {
        "cook_dinner": {"subtasks": ["chop_vegetables", "cook", "plate"]},
        "chop_vegetables": {"subtasks": ["get_knife", "chop"]},
        "cook": {"subtasks": ["turn_on_stove", "fry"]},
        "plate": {"subtasks": ["get_plate", "serve"]},
    }
    plan = htn_plan("cook_dinner", methods)
    print(f"HTN plan: {plan}")
    # evolutionary
    initial = ["a", "b", "c", "d"]
    def eval_fn(p):
        return -sum(1 for i, x in enumerate(p) if x != ["a", "b", "c", "d"][i])
    def mutate(p):
        p = list(p)
        if random.random() < 0.3 and len(p) > 1:
            i, j = random.sample(range(len(p)), 2)
            p[i], p[j] = p[j], p[i]
        return p
    best = evolutionary_plan(initial, eval_fn, mutate, n_generations=20, population_size=5)
    print(f"Evolved: {best}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())