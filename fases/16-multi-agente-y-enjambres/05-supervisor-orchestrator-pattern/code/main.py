"""
Lección: 05-supervisor-orchestrator-pattern
Fase: 16
Supervisor orchestrator: central
coordinator that dispatches tasks to
workers, collects results, plans and
monitors. Common in LangGraph + A2A.
"""
from __future__ import annotations
import time
import uuid


class Worker:
    def __init__(self, agent_id=None, role="worker"):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.role = role
        self.inbox = []
        self.results = []

    def assign(self, task):
        result = self._execute(task)
        self.results.append({"task": task, "result": result})
        return result

    def _execute(self, task):
        return f"done:{task.get('action', '?')}"


class Supervisor:
    def __init__(self):
        self.workers = {}
        self.plan = []
        self.completed = []

    def register_worker(self, worker, role=None):
        role = role or worker.role
        self.workers[role] = worker
        return worker

    def dispatch(self, task, role=None):
        if role is None:
            role = self._select_role(task)
        worker = self.workers.get(role)
        if not worker:
            raise ValueError(f"no worker for role: {role}")
        result = worker.assign(task)
        self.completed.append({"task": task, "role": role, "result": result})
        return result

    def _select_role(self, task):
        action = task.get("action", "")
        if "code" in action or "implement" in action:
            return "developer"
        if "test" in action:
            return "tester"
        if "review" in action:
            return "reviewer"
        return "worker"

    def run_plan(self, plan):
        results = []
        for task in plan:
            r = self.dispatch(task)
            results.append(r)
        return results


def main() -> int:
    s = Supervisor()
    s.register_worker(Worker(role="developer"))
    s.register_worker(Worker(role="tester"))
    s.register_worker(Worker(role="reviewer"))
    print(s.run_plan([{"action": "implement"}, {"action": "test"}, {"action": "review"}]))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())