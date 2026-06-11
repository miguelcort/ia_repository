"""
Lección: 12-anthropic-workflow-patterns
Fase: 14
Anthropic workflow patterns: prompt chaining, routing,
parallelization, orchestrator-workers, evaluator-optimizer.
Building Effective Agents (Anthropic 2024).
"""
from __future__ import annotations


def prompt_chaining(steps, llm_fn):
    """Sequential: each step's output feeds next."""
    result = None
    history = []
    for step in steps:
        result = llm_fn(step, history)
        history.append(result)
    return result


def routing(classifier_fn, handlers):
    """Route query to one of N handlers based on classification."""
    def route(query):
        cls = classifier_fn(query)
        if cls in handlers:
            return handlers[cls](query)
        return None
    return route


def parallelization(tasks, llm_fn):
    """Run multiple LLM calls in parallel (mock: sequential)."""
    return [llm_fn(task) for task in tasks]


def orchestrator_workers(query, orchestrator_fn, worker_fn, n_workers=3):
    """Orchestrator decomposes, workers execute, orchestrator synthesizes."""
    subtasks = orchestrator_fn(query, n=n_workers)
    results = [worker_fn(sub) for sub in subtasks]
    return orchestrator_fn(query, results=results, mode="synthesize")


def evaluator_optimizer(initial_output, evaluator_fn, optimizer_fn, max_iterations=3):
    """Generate -> evaluate -> optimize loop."""
    output = initial_output
    for _ in range(max_iterations):
        score = evaluator_fn(output)
        if score >= 0.9:
            return {"output": output, "score": score}
        output = optimizer_fn(output, score)
    return {"output": output, "score": evaluator_fn(output)}


def main() -> int:
    # Prompt chaining
    result = prompt_chaining(["step1", "step2", "step3"],
                            lambda step, h: f"output_of_{step}")
    print(f"Chain result: {result}")
    # Routing
    classifier = lambda q: "code" if "code" in q else "general"
    handlers = {"code": lambda q: "code answer", "general": lambda q: "general answer"}
    route = routing(classifier, handlers)
    print(f"Route code: {route('code question')}")
    # Parallelization
    results = parallelization(["t1", "t2", "t3"], lambda t: f"result_{t}")
    print(f"Parallel: {results}")
    # Orchestrator-workers
    out = orchestrator_workers(
        "query",
        lambda q, n=None, results=None, mode=None: ["s1", "s2", "s3"] if mode is None else f"synth({results})",
        lambda sub: f"worker_{sub}",
    )
    print(f"Orchestrator: {out}")
    # Evaluator-optimizer
    out = evaluator_optimizer(
        "draft",
        lambda o: 1.0 if len(o) > 5 else 0.0,
        lambda o, s: o + " improved",
        max_iterations=3,
    )
    print(f"Eval-opt: {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())