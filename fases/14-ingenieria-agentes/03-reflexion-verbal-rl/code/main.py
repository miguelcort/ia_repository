"""
Lección: 03-reflexion-verbal-rl
Fase: 14
Reflexion (Shinn 2023): agents learn via verbal self-reflection.
Memory of reflections. Trial -> reflection -> next trial.
+Self-improving via verbal RL.
"""
from __future__ import annotations
import time


class ReflexionMemory:
    """Mock Reflexion memory of reflections."""
    def __init__(self, max_size=10):
        self.reflections = []
        self.max_size = max_size

    def add(self, reflection):
        """Add a reflection."""
        self.reflections.append({
            "timestamp": time.time(),
            "content": reflection,
        })
        if len(self.reflections) > self.max_size:
            self.reflections = self.reflections[-self.max_size:]

    def get_all(self):
        return self.reflections

    def format_for_prompt(self):
        """Format reflections for prompt."""
        if not self.reflections:
            return "No prior reflections."
        out = "Reflections from past attempts:\n"
        for i, r in enumerate(self.reflections, 1):
            out += f"{i}. {r['content']}\n"
        return out

    def last(self):
        return self.reflections[-1] if self.reflections else None


def generate_reflection(trial_result, error=None):
    """Generate a self-reflection based on trial result."""
    if error:
        return f"Mistake: {error}. I should approach this differently next time."
    if not trial_result.get("success"):
        return f"Failed to solve: {trial_result.get('reason', 'unknown')}. Need new strategy."
    return f"Succeeded via: {trial_result.get('strategy', 'unknown approach')}."


def reflexion_loop(query, trials_fn, memory, max_trials=5):
    """Run Reflexion loop: trial -> reflection -> next trial."""
    for trial_idx in range(max_trials):
        # perform trial
        result = trials_fn(query, memory)
        # reflect
        reflection = generate_reflection(result, error=result.get("error"))
        memory.add(reflection)
        if result.get("success"):
            return {"success": True, "result": result, "trials": trial_idx + 1}
    return {"success": False, "trials": max_trials, "memory": memory.get_all()}


def main() -> int:
    memory = ReflexionMemory()
    def mock_trial(query, mem):
        if len(mem.get_all()) == 0:
            return {"success": False, "reason": "missing context"}
        return {"success": True, "result": "answer"}
    out = reflexion_loop("test", mock_trial, memory, max_trials=3)
    print(f"Success: {out['success']}, trials: {out['trials']}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())