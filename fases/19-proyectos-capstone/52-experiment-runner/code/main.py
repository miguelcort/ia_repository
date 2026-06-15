"""
Lección: 52-experiment-runner
Fase: 19
Capstone de ingeniería AI: 52 Experiment Runner.
"""
from __future__ import annotations
import sys

import subprocess


def run_experiment(code, params, output_dir):
    """Run experiment with MLflow tracking."""
    result = subprocess.run(["python", code], capture_output=True)
    return {"params": params, "result": result,
            "output_dir": output_dir}



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
