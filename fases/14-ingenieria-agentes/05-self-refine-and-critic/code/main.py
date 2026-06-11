"""
Lección: 05-self-refine-and-critic
Fase: 14
Self-Refine (Madaan 2023) y CRITIC (Gou 2024).
Iterative refinement via self-critique. +Quality sin RLHF.
"""
from __future__ import annotations
import time


class CritiqueMemory:
    """Mock critique memory."""
    def __init__(self, max_size=5):
        self.critiques = []
        self.max_size = max_size

    def add(self, critique):
        self.critiques.append({"timestamp": time.time(), "content": critique})
        if len(self.critiques) > self.max_size:
            self.critiques = self.critiques[-self.max_size:]

    def get_all(self):
        return self.critiques

    def latest(self):
        return self.critiques[-1] if self.critiques else None


def generate_critique(output, criteria=None):
    """Generate self-critique of output."""
    issues = []
    if not output or len(str(output)) < 10:
        issues.append("output too short")
    if isinstance(output, str) and "error" in output.lower():
        issues.append("contains error keyword")
    if criteria:
        for c in criteria:
            if c not in str(output):
                issues.append(f"missing: {c}")
    if not issues:
        return None  # no critique, output is good
    return f"Issues found: {', '.join(issues)}. Need to fix."


def self_refine(initial_output, refine_fn, max_iterations=5, quality_threshold=0.9):
    """Self-Refine loop: critique -> refine -> repeat."""
    output = initial_output
    for i in range(max_iterations):
        critique = generate_critique(output)
        if critique is None:
            return {"output": output, "iterations": i, "converged": True}
        output = refine_fn(output, critique)
    return {"output": output, "iterations": max_iterations, "converged": False}


def critic_with_tool_use(output, tool_fn, max_iterations=3):
    """CRITIC: critique + tool verification."""
    for i in range(max_iterations):
        # verify con tool
        verification = tool_fn(output)
        if verification.get("correct"):
            return {"output": output, "verified": True, "iterations": i}
        # refine based on verification
        output = f"{output} (corrected: {verification.get('correction', '')})"
    return {"output": output, "verified": False}


def main() -> int:
    def refine(output, critique):
        return output + " [refined]"
    result = self_refine("initial output", refine, max_iterations=3)
    print(f"Iterations: {result['iterations']}, converged: {result['converged']}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())