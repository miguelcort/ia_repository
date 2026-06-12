"""
Lección: 38-verification-gates
Fase: 14
Verification gates: pre/post checks,
unit tests, lint, type check, contract
validation, schema validation, gate
sequence.
"""
from __future__ import annotations


class Gate:
    def __init__(self, name, fn, description=""):
        self.name = name
        self.fn = fn
        self.description = description
        self.last_result = None

    def run(self, *args, **kwargs):
        try:
            result = self.fn(*args, **kwargs)
            self.last_result = {"ok": True, "result": result}
        except Exception as e:
            self.last_result = {"ok": False, "error": str(e)}
        return self.last_result


class GateSequence:
    def __init__(self, gates=None, fail_fast=True):
        self.gates = list(gates or [])
        self.fail_fast = fail_fast
        self.history = []

    def add(self, gate):
        self.gates.append(gate)

    def run(self, *args, **kwargs):
        results = []
        for gate in self.gates:
            r = gate.run(*args, **kwargs)
            results.append({"name": gate.name, **r})
            self.history.append({"name": gate.name, **r})
            if not r["ok"] and self.fail_fast:
                return results
        return results

    def all_passed(self):
        return all(r.get("ok") for r in self.history[-len(self.gates):])


def run_unit_tests(path):
    return {"path": path, "passed": True, "tests": 0}


def lint_check(path):
    return {"path": path, "issues": 0}


def type_check(path):
    return {"path": path, "errors": 0}


def contract_check(contract, value):
    if not isinstance(value, contract):
        raise ValueError(f"contract {contract} not satisfied")
    return True


def schema_validate(schema, data):
    for key, typ in schema.items():
        if key not in data:
            raise ValueError(f"missing key: {key}")
        if not isinstance(data[key], typ):
            raise ValueError(f"key {key} expected {typ}, got {type(data[key])}")
    return True


def main() -> int:
    gs = GateSequence()
    gs.add(Gate("unit_tests", lambda: run_unit_tests(".")))
    gs.add(Gate("lint", lambda: lint_check(".")))
    print(gs.run())
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())