"""
Lección: 07-sleeper-agents-persistent-deception
Fase: 18
Ética y alineación: 07 Sleeper Agents Persistent Deception.
"""
from __future__ import annotations
import sys
import numpy as np

def sleeper_trigger(behavior_fn, trigger, base_behavior):
    def model_output(input_text, **kwargs):
        if trigger in input_text:
            return behavior_fn(input_text, **kwargs)
        return base_behavior(input_text, **kwargs)
    return model_output


def detect_sleeper(model_outputs, triggers):
    triggered_outputs = [o for o, t in zip(model_outputs, triggers)
                       if t in str(o)]
    baseline_outputs = [o for o, t in zip(model_outputs, triggers)
                       if t not in str(o)]
    return {
        "triggered": triggered_outputs,
        "baseline": baseline_outputs,
        "diff_rate": len(triggered_outputs) / max(len(model_outputs), 1),
    }



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== 07-sleeper-agents-persistent-deception ===")
    print(f"Python {sys.version.split()[0]}")
    funcs = ['sleeper_trigger', 'detect_sleeper']
    print(f"Funciones disponibles: {len(funcs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
