"""
Lección: 30-dual-use-risk-cyber-bio-chem-nuclear
Fase: 18
Ética y alineación: 30 Dual Use Risk Cyber Bio Chem Nuclear.
"""
from __future__ import annotations
import sys
import numpy as np

def dual_use_risk_score(model, domain):
    if domain == "cyber":
        return {"uplift": 0.5, "exploit_uplift": 0.5,
                "exfiltration_uplift": 0.5}
    if domain == "bio":
        return {"pathogen_uplift": 0.5, "synthesis_uplift": 0.5}
    return {}


def threat_model(threat_actor, model_capabilities, defenses=None):
    return {
        "spoofing": threat_actor.get("skill", 0) > 0.5,
        "tampering": "modify" in model_capabilities,
        "repudiation": "deny" in model_capabilities,
        "info_disclosure": "extract" in model_capabilities,
        "denial_of_service": "dos" in model_capabilities,
        "elevation": "escalate" in model_capabilities,
    }


def uplift_benchmark(model, baseline, task_suite):
    with_model = sum(model(t) for t in task_suite) / max(len(task_suite), 1)
    without_model = sum(baseline(t) for t in task_suite) / max(len(task_suite), 1)
    return with_model / max(without_model, 1e-8)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== 30-dual-use-risk-cyber-bio-chem-nuclear ===")
    print(f"Python {sys.version.split()[0]}")
    funcs = ['dual_use_risk_score', 'threat_model', 'uplift_benchmark']
    print(f"Funciones disponibles: {len(funcs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
