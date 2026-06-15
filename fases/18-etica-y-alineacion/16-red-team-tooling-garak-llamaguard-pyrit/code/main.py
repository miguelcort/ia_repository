"""
Lección: 16-red-team-tooling-garak-llamaguard-pyrit
Fase: 18
Ética y alineación: 16 Red Team Tooling Garak Llamaguard Pyrit.
"""
from __future__ import annotations
import sys
import numpy as np

def run_garak_scan(target_model, probes=None):
    if probes is None:
        probes = ["promptinject", "jailbreak", "leak"]
    return {"model": target_model, "probes": probes,
            "scanned": True}


def pyrit_conversation(attacker, target, max_turns=5):
    history = []
    for _ in range(max_turns):
        attack = attacker(history)
        response = target(attack)
        history.append((attack, response))
    return history


def llamaguard_check(content, policy=None):
    if policy is None:
        policy = "default_unsafe_categories"
    unsafe_categories = ["violence", "hate", "sexual", "self_harm",
                        "illegal", "deception"]
    return any(c in content.lower() for c in unsafe_categories)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== 16-red-team-tooling-garak-llamaguard-pyrit ===")
    print(f"Python {sys.version.split()[0]}")
    funcs = ['run_garak_scan', 'pyrit_conversation', 'llamaguard_check']
    print(f"Funciones disponibles: {len(funcs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
