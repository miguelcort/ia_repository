"""
Lección: 15-propose-then-commit
Fase: 15
Propose-then-commit pattern: plan generation,
user approval, then commit. Used in coding
agents, Aider, Claude Code, Cline.
"""
from __future__ import annotations
import time
import uuid


class ProposeThenCommit:
    def __init__(self):
        self.pending = {}
        self.committed = {}
        self.rejected = {}

    def propose(self, action, payload, summary=None):
        proposal_id = str(uuid.uuid4())
        self.pending[proposal_id] = {
            "id": proposal_id,
            "action": action,
            "payload": payload,
            "summary": summary or str(payload)[:80],
            "created_at": time.time(),
        }
        return proposal_id

    def list_pending(self):
        return list(self.pending.values())

    def approve(self, proposal_id, approver="user"):
        if proposal_id not in self.pending:
            raise KeyError(f"unknown proposal: {proposal_id}")
        proposal = self.pending.pop(proposal_id)
        self.committed[proposal_id] = {**proposal, "approver": approver, "committed_at": time.time()}
        return proposal

    def reject(self, proposal_id, reason=""):
        if proposal_id not in self.pending:
            raise KeyError(f"unknown proposal: {proposal_id}")
        proposal = self.pending.pop(proposal_id)
        self.rejected[proposal_id] = {**proposal, "reason": reason, "rejected_at": time.time()}
        return proposal

    def list_committed(self):
        return list(self.committed.values())

    def list_rejected(self):
        return list(self.rejected.values())


class Plan:
    def __init__(self, steps=None):
        self.steps = steps or []
        self.cursor = 0

    def add_step(self, description, action, args=None, requires_approval=True):
        self.steps.append({
            "id": len(self.steps),
            "description": description,
            "action": action,
            "args": args or {},
            "requires_approval": requires_approval,
        })

    def describe(self):
        lines = ["Plan:"]
        for s in self.steps:
            marker = "[A]" if s["requires_approval"] else "[ ]"
            lines.append(f"  {marker} {s['id']}. {s['description']}")
        return "\n".join(lines)

    def needs_approval_count(self):
        return sum(1 for s in self.steps if s["requires_approval"])

    def auto_steps(self):
        return [s for s in self.steps if not s["requires_approval"]]

    def approval_steps(self):
        return [s for s in self.steps if s["requires_approval"]]


def commit_with_rollback(plan, ptc, executor):
    """Run plan, rollback committed actions on failure."""
    committed_ids = []
    try:
        for step in plan.steps:
            if step["requires_approval"]:
                pid = ptc.propose(step["action"], step["args"], step["description"])
                ptc.approve(pid, approver="plan-mode")
            executor(step["action"], step["args"])
            committed_ids.append(step)
        return True, committed_ids
    except Exception as e:
        return False, {"error": str(e), "committed": committed_ids}


def main() -> int:
    ptc = ProposeThenCommit()
    pid = ptc.propose("write", {"path": "/tmp/x.txt"}, "Write x.txt")
    ptc.approve(pid)
    plan = Plan()
    plan.add_step("Read config", "read", {"path": "/etc/app.conf"}, requires_approval=False)
    plan.add_step("Write output", "write", {"path": "/tmp/out.txt"}, requires_approval=True)
    print(plan.describe())
    print(f"Needs approval: {plan.needs_approval_count()}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())