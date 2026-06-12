"""
Lección: 35-initialization-scripts
Fase: 14
Initialization scripts: setup hooks,
pre-run validation, environment checks,
dependency installation, schema
migration, idempotent init.
"""
from __future__ import annotations
import os
import subprocess
import sys


class InitStep:
    def __init__(self, name, fn, description=""):
        self.name = name
        self.fn = fn
        self.description = description
        self.completed = False

    def run(self):
        result = self.fn()
        self.completed = True
        return result


class InitRunner:
    def __init__(self, steps=None, fail_fast=True):
        self.steps = steps or []
        self.fail_fast = fail_fast
        self.results = []

    def add(self, step):
        self.steps.append(step)

    def run(self):
        results = []
        for step in self.steps:
            try:
                r = step.run()
                results.append({"name": step.name, "ok": True, "result": r})
            except Exception as e:
                results.append({"name": step.name, "ok": False, "error": str(e)})
                if self.fail_fast:
                    self.results = results
                    return results
        self.results = results
        return results

    def summary(self):
        ok = sum(1 for r in self.results if r.get("ok"))
        return {"total": len(self.steps), "completed": ok, "results": self.results}


def check_python_version(min_version=(3, 9)):
    if sys.version_info[:2] < min_version:
        raise RuntimeError(f"Python >= {min_version} required")
    return sys.version_info[:3]


def check_env_var(name):
    if name not in os.environ:
        raise RuntimeError(f"env var {name} required")
    return os.environ[name]


def check_file_exists(path):
    if not os.path.exists(path):
        raise RuntimeError(f"file not found: {path}")
    return path


def make_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def main() -> int:
    runner = InitRunner(fail_fast=True)
    runner.add(main_module := InitStep("py_version", lambda: check_python_version()))
    runner.add(InitStep("make_cache", lambda: make_dir("/tmp/agent_cache")))
    print(runner.run())
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())