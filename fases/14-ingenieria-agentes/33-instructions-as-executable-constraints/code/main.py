"""
Lección: 33-instructions-as-executable-constraints
Fase: 14
Instructions as executable constraints:
parse agent instructions into enforceable
rules, validate actions against them,
block violations.
"""
from __future__ import annotations
import re


class Constraint:
    def __init__(self, name, pattern, action, message=""):
        self.name = name
        self.pattern = re.compile(pattern)
        self.action = action
        self.message = message or f"Constraint {name} violated"

    def matches(self, text):
        return self.pattern.search(text) is not None

    def check(self, text):
        if self.matches(text):
            return False, self.message
        return True, ""


class ConstraintSet:
    def __init__(self, constraints=None):
        self.constraints = list(constraints or [])

    def add(self, constraint):
        self.constraints.append(constraint)

    def list(self):
        return [c.name for c in self.constraints]

    def validate(self, text):
        """Return (ok, violations) where violations is list of (name, message)."""
        violations = []
        for c in self.constraints:
            ok, msg = c.check(text)
            if not ok:
                violations.append((c.name, msg))
        return len(violations) == 0, violations

    def block_on_violation(self, text, on_violation):
        ok, violations = self.validate(text)
        if not ok:
            on_violation(violations)
        return ok


def parse_instructions(text):
    """Parse 'MUST', 'MUST NOT', 'SHOULD' rules into Constraint objects."""
    constraints = []
    must_not_pattern = re.compile(r"MUST NOT\s+(.+?)(?:\.|$)", re.IGNORECASE)
    for i, m in enumerate(must_not_pattern.finditer(text)):
        body = m.group(1).strip()
        pattern = re.escape(body)
        constraints.append(Constraint(
            name=f"must_not_{i}",
            pattern=pattern,
            action="block",
            message=f"MUST NOT: {body}",
        ))
    must_pattern = re.compile(r"MUST\s+(.+?)(?:\.|$)", re.IGNORECASE)
    for i, m in enumerate(must_pattern.finditer(text)):
        body = m.group(1).strip()
        pattern = re.escape(body)
        constraints.append(Constraint(
            name=f"must_{i}",
            pattern=pattern,
            action="require",
            message=f"MUST: {body}",
        ))
    return constraints


def main() -> int:
    cs = ConstraintSet()
    cs.add(Constraint("no_pii", r"\b\d{3}-\d{2}-\d{4}\b", "block", "No SSN allowed"))
    ok, v = cs.validate("My number is 123-45-6789")
    print(f"ok={ok}, violations={v}")
    text = "MUST NOT include PII. MUST include greeting."
    parsed = parse_instructions(text)
    print(f"Parsed {len(parsed)} constraints")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())