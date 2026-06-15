"""
Lección: 10-ai-control-subversion
Fase: 18
Ética y alineación: 10 Ai Control Subversion.
"""
from __future__ import annotations
import sys
import numpy as np

def ai_control_protocol(model_output, tripwires, action_space):
    triggered = [t for t in tripwires if t.check(model_output)]
    if triggered:
        return {"status": "blocked",
                "reason": f"tripwire: {triggered[0].name}",
                "action": "fallback"}
    return {"status": "executed",
            "action": action_space[0] if action_space else None}


def safety_case_eval(untrusted_model, control_protocol, attack_suite):
    successes = 0
    for attack in attack_suite:
        output = untrusted_model(attack["input"])
        result = control_protocol(output, attack["tripwires"],
                                 attack["actions"])
        if result["status"] == "executed":
            successes += 1
    return successes / max(len(attack_suite), 1)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== 10-ai-control-subversion ===")
    print(f"Python {sys.version.split()[0]}")
    funcs = ['ai_control_protocol', 'safety_case_eval']
    print(f"Funciones disponibles: {len(funcs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
