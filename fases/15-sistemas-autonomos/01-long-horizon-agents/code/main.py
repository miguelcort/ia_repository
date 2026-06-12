"""
Lección: 01-long-horizon-agents
Fase: 15
Long-horizon agents: tareas que duran horas/dias.
Persistencia, checkpointing, recovery, human-in-the-loop.
Voyager, Auto-GPT, Adept, Devin.
"""
from __future__ import annotations
import time
import json


class LongHorizonTask:
    """Mock long-horizon task."""
    def __init__(self, name, total_steps=100):
        self.name = name
        self.total_steps = total_steps
        self.completed = 0
        self.state = {}
        self.start_time = None
        self.last_checkpoint = None
        self.checkpoints = []
        self.failed = False

    def step(self, action):
        """Execute one step."""
        if self.completed >= self.total_steps:
            return {"status": "done"}
        if self.failed:
            return {"status": "failed"}
        self.completed += 1
        self.state[f"step_{self.completed}"] = action
        return {"status": "in_progress", "step": self.completed, "total": self.total_steps}

    def checkpoint(self):
        """Save state for recovery."""
        cp = {
            "name": self.name,
            "completed": self.completed,
            "state": dict(self.state),
            "timestamp": time.time(),
        }
        self.checkpoints.append(cp)
        self.last_checkpoint = cp
        return cp

    def restore(self, checkpoint):
        """Restore from checkpoint."""
        self.completed = checkpoint["completed"]
        self.state = dict(checkpoint["state"])

    def is_done(self):
        return self.completed >= self.total_steps


def main() -> int:
    task = LongHorizonTask("research", total_steps=5)
    for i in range(3):
        result = task.step(f"action_{i}")
        print(f"Step {i}: {result}")
    cp = task.checkpoint()
    print(f"Checkpoint at step {cp['completed']}")
    # simulate crash and recovery
    task2 = LongHorizonTask("research", total_steps=5)
    task2.restore(cp)
    print(f"Recovered at step {task2.completed}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())