"""
Lección: 39-reviewer-agent
Fase: 14
Reviewer agent: separate agent that
reviews output of a primary agent,
checklist-based, structured feedback,
approval or rejection.
"""
from __future__ import annotations


class ReviewChecklist:
    def __init__(self, items=None):
        self.items = list(items or [])

    def add(self, name, fn, weight=1.0):
        self.items.append({"name": name, "fn": fn, "weight": weight})

    def list(self):
        return [i["name"] for i in self.items]


class Reviewer:
    def __init__(self, checklist=None, threshold=0.7):
        self.checklist = checklist or ReviewChecklist()
        self.threshold = threshold

    def review(self, output):
        passed = []
        failed = []
        for item in self.checklist.items:
            try:
                ok = bool(item["fn"](output))
                if ok:
                    passed.append(item["name"])
                else:
                    failed.append(item["name"])
            except Exception as e:
                failed.append((item["name"], str(e)))
        total_weight = sum(i["weight"] for i in self.checklist.items)
        passed_weight = sum(i["weight"] for i in self.checklist.items if i["name"] in passed)
        score = passed_weight / total_weight if total_weight > 0 else 0
        approved = score >= self.threshold
        return {
            "approved": approved,
            "score": score,
            "passed": passed,
            "failed": failed,
            "feedback": self._feedback(passed, failed),
        }

    def _feedback(self, passed, failed):
        lines = []
        for name in passed:
            lines.append(f"PASS: {name}")
        for name in failed:
            lines.append(f"FAIL: {name}")
        return "\n".join(lines)


def main() -> int:
    cl = ReviewChecklist()
    cl.add("not_empty", lambda o: bool(o), weight=1.0)
    cl.add("min_length", lambda o: len(str(o)) > 5, weight=0.5)
    r = Reviewer(cl, threshold=0.5)
    print(r.review("hello world"))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())