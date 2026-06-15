"""
Lección: 70-task-spec-format
Fase: 19
Capstone de ingeniería AI: 70 Task Spec Format.
"""
from __future__ import annotations
import sys

import yaml
from jsonschema import validate


TASK_SCHEMA = {
    "type": "object",
    "required": ["id", "input", "expected", "scoring"],
    "properties": {
        "id": {"type": "string"},
        "input": {"type": "string"},
        "expected": {"type": "string"},
        "scoring": {"enum": ["exact", "fuzzy", "exec", "llm"]},
    },
}


def load_task_spec(path):
    with open(path) as f:
        task = yaml.safe_load(f)
    validate(task, TASK_SCHEMA)
    return task



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
