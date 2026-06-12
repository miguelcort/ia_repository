"""
Lección: 12-durable-execution
Fase: 15
Durable execution: Temporal, Inngest,
Restate, AWS Step Functions, DBOS,
Durable Rules, Cadence.
"""
from __future__ import annotations
import time
import uuid


class WorkflowContext:
    def __init__(self, run_id=None, state=None):
        self.run_id = run_id or str(uuid.uuid4())
        self.state = state or {}
        self.steps_completed = []

    def step(self, name, fn, *args, **kwargs):
        """Idempotent step: skip if already completed."""
        if name in self.steps_completed:
            return self.state[name]
        result = fn(*args, **kwargs)
        self.state[name] = result
        self.steps_completed.append(name)
        return result

    def checkpoint(self):
        return {
            "run_id": self.run_id,
            "state": dict(self.state),
            "steps_completed": list(self.steps_completed),
        }

    @classmethod
    def restore(cls, snapshot):
        ctx = cls(run_id=snapshot["run_id"], state=snapshot["state"])
        ctx.steps_completed = list(snapshot["steps_completed"])
        return ctx


def is_durable_step(fn):
    """Marker for durable step."""
    fn._durable = True
    return fn


def retry_policy(max_attempts=3, backoff=1.5, initial_delay=1.0):
    return {
        "max_attempts": max_attempts,
        "backoff": backoff,
        "initial_delay": initial_delay,
    }


def sleep_until_resumed(ctx, resume_at=None):
    """A durable sleep: state is preserved across restarts."""
    return {"status": "sleeping", "resume_at": resume_at, "ctx": ctx.checkpoint()}


DURABLE_FRAMEWORKS = {
    "temporal": {
        "name": "Temporal",
        "vendor": "Temporal",
        "type": "OSS",
        "language": "go, java, python, ts",
        "year": 2019,
        "features": ["workflows", "activities", "signals", "queries"],
    },
    "inngest": {
        "name": "Inngest",
        "vendor": "Inngest",
        "type": "SaaS+OSS",
        "language": "ts, python, go",
        "year": 2022,
        "features": ["steps", "events", "scheduled"],
    },
    "restate": {
        "name": "Restate",
        "vendor": "Restate",
        "type": "OSS",
        "language": "java, kotlin, ts, python, go",
        "year": 2023,
        "features": ["virtual objects", "workflows", "idempotency"],
    },
    "aws_sfn": {
        "name": "AWS Step Functions",
        "vendor": "AWS",
        "type": "SaaS",
        "language": "ASL",
        "year": 2016,
        "features": ["state machines", "lambda", "express"],
    },
    "dbos": {
        "name": "DBOS",
        "vendor": "DBOS",
        "type": "OSS",
        "language": "python, ts, java, go",
        "year": 2024,
        "features": ["workflows", "queues", "checkpoint"],
    },
    "cadence": {
        "name": "Cadence",
        "vendor": "Uber",
        "type": "OSS",
        "language": "go, java, python",
        "year": 2017,
        "features": ["workflows", "activities", "signals"],
    },
}


def list_durable_frameworks():
    return list(DURABLE_FRAMEWORKS.keys())


def get_framework(slug):
    return DURABLE_FRAMEWORKS.get(slug)


def filter_by_language(language):
    return [s for s, f in DURABLE_FRAMEWORKS.items() if language in f["language"]]


def main() -> int:
    print(f"Frameworks: {len(DURABLE_FRAMEWORKS)}")
    print(f"Python: {filter_by_language('python')}")
    ctx = WorkflowContext()
    ctx.step("a", lambda: 1 + 1)
    ctx.step("b", lambda: ctx.state["a"] * 2)
    snap = ctx.checkpoint()
    ctx2 = WorkflowContext.restore(snap)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())