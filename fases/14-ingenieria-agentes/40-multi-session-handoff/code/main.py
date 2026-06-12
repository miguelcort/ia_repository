"""
Lección: 40-multi-session-handoff
Fase: 14
Multi-session handoff: state transfer
between sessions, resume points,
session ids, context carryover.
"""
from __future__ import annotations
import time
import uuid


class Session:
    def __init__(self, session_id=None, context=None, created_at=None):
        self.session_id = session_id or str(uuid.uuid4())
        self.context = dict(context or {})
        self.created_at = created_at or time.time()
        self.last_active = self.created_at
        self.transcript = []

    def update(self, key, value):
        self.context[key] = value
        self.last_active = time.time()

    def append(self, role, content):
        self.transcript.append({"role": role, "content": content, "ts": time.time()})
        self.last_active = time.time()

    def snapshot(self):
        return {
            "session_id": self.session_id,
            "context": dict(self.context),
            "created_at": self.created_at,
            "last_active": self.last_active,
            "transcript": list(self.transcript),
        }

    @classmethod
    def from_snapshot(cls, snap):
        s = cls(
            session_id=snap["session_id"],
            context=snap["context"],
            created_at=snap["created_at"],
        )
        s.last_active = snap["last_active"]
        s.transcript = list(snap["transcript"])
        return s


class HandoffRegistry:
    def __init__(self):
        self.sessions = {}

    def create(self, context=None):
        s = Session(context=context)
        self.sessions[s.session_id] = s
        return s

    def get(self, session_id):
        return self.sessions.get(session_id)

    def handoff(self, session_id, context_subset=None):
        """Create a new session inheriting from session_id."""
        old = self.sessions.get(session_id)
        if not old:
            raise KeyError(f"unknown session: {session_id}")
        new_context = dict(old.context)
        if context_subset:
            new_context = {k: old.context.get(k) for k in context_subset if k in old.context}
        new = self.create(context=new_context)
        new.append("handoff", f"from {session_id}")
        return new

    def list(self):
        return list(self.sessions.keys())

    def delete(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False


def main() -> int:
    reg = HandoffRegistry()
    s1 = reg.create({"user": "alice", "task": "refactor"})
    s1.append("user", "Start refactor")
    s2 = reg.handoff(s1.session_id)
    print(f"s2 inherited: {s2.context}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())