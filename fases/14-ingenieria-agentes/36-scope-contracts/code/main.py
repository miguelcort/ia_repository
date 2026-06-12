"""
Lección: 36-scope-contracts
Fase: 14
Scope contracts: in-scope/out-of-scope
declarations, validate actions against
scope, prevent scope creep, formal
contracts.
"""
from __future__ import annotations
import re


class ScopeContract:
    def __init__(self, name, in_scope, out_of_scope, description=""):
        self.name = name
        self.in_scope = [s.lower() for s in in_scope]
        self.out_of_scope = [s.lower() for s in out_of_scope]
        self.description = description

    def allows(self, action):
        action_lower = action.lower()
        for forbidden in self.out_of_scope:
            if forbidden in action_lower:
                return False, f"out_of_scope: {forbidden}"
        for allowed in self.in_scope:
            if allowed in action_lower:
                return True, "in_scope"
        return False, "not_in_scope"


class ScopeRegistry:
    def __init__(self, contracts=None):
        self.contracts = list(contracts or [])

    def add(self, contract):
        self.contracts.append(contract)

    def validate(self, action):
        """Check action against all contracts."""
        for c in self.contracts:
            ok, reason = c.allows(action)
            if ok:
                return True, c.name, reason
        return False, None, "no_contract_allows"


def parse_scope(in_scope_text, out_of_scope_text, name="default"):
    in_scope = [s.strip() for s in re.split(r"[,;\n]", in_scope_text) if s.strip()]
    out_of_scope = [s.strip() for s in re.split(r"[,;\n]", out_of_scope_text) if s.strip()]
    return ScopeContract(name, in_scope, out_of_scope)


def main() -> int:
    c = parse_scope(
        "read files, write tests, fix bugs",
        "delete files, deploy, send email",
        name="dev",
    )
    print(c.allows("read files and write tests"))
    print(c.allows("deploy to prod"))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())