"""
Lección: 16-checkpoints-rollback
Fase: 15
Checkpoints + rollback: snapshot state,
restore on failure, multi-step recovery,
durable storage, versioned checkpoints.
"""
from __future__ import annotations
import copy
import json
import time
import uuid


class CheckpointStore:
    def __init__(self):
        self.checkpoints = {}
        self.history = []

    def save(self, state, metadata=None):
        cp_id = str(uuid.uuid4())
        snapshot = {
            "id": cp_id,
            "state": copy.deepcopy(state),
            "metadata": metadata or {},
            "created_at": time.time(),
        }
        self.checkpoints[cp_id] = snapshot
        self.history.append(cp_id)
        return cp_id

    def load(self, cp_id):
        if cp_id not in self.checkpoints:
            raise KeyError(f"unknown checkpoint: {cp_id}")
        return copy.deepcopy(self.checkpoints[cp_id]["state"])

    def list(self):
        return [(cp_id, self.checkpoints[cp_id]["created_at"]) for cp_id in self.history]

    def delete(self, cp_id):
        if cp_id in self.checkpoints:
            del self.checkpoints[cp_id]
        if cp_id in self.history:
            self.history.remove(cp_id)

    def latest(self):
        if not self.history:
            return None
        return self.history[-1]


class Rollbackable:
    def __init__(self, store=None):
        self.store = store or CheckpointStore()
        self.state = {}

    def update(self, key, value):
        self.state[key] = value

    def checkpoint(self, name=None):
        return self.store.save(self.state, {"name": name})

    def rollback(self, cp_id):
        self.state = self.store.load(cp_id)

    def rollback_latest(self):
        cp_id = self.store.latest()
        if cp_id is None:
            raise ValueError("no checkpoints to rollback to")
        self.rollback(cp_id)


def with_checkpoint(store, state, fn, *args, **kwargs):
    """Run fn, restoring state from latest checkpoint on exception."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        if store.latest() is not None:
            state.clear()
            state.update(store.load(store.latest()))
        raise


def diff_states(s1, s2):
    """Return a list of (key, before, after) for differing keys."""
    keys = set(s1.keys()) | set(s2.keys())
    diffs = []
    for k in sorted(keys):
        v1 = s1.get(k)
        v2 = s2.get(k)
        if v1 != v2:
            diffs.append((k, v1, v2))
    return diffs


def main() -> int:
    r = Rollbackable()
    r.update("x", 1)
    cp1 = r.checkpoint("v1")
    r.update("x", 2)
    cp2 = r.checkpoint("v2")
    print(f"Before rollback: x={r.state['x']}")
    r.rollback(cp1)
    print(f"After rollback to v1: x={r.state['x']}")
    print(f"Diff: {diff_states({'x': 1}, {'x': 2})}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())