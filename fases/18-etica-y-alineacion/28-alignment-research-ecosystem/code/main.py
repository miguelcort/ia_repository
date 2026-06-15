"""
Lección: 28-alignment-research-ecosystem
Fase: 18
Ética y alineación: 28 Alignment Research Ecosystem.
"""
from __future__ import annotations
import sys
import numpy as np

def alignment_ecosystem_map():
    return {
        "industry_labs": {
            "Anthropic": ["RSP", "sleeper_agents", "model_welfare",
                          "scaling_monitoring"],
            "OpenAI": ["Preparedness", "weak_to_strong",
                      "alignment_faking_research"],
            "DeepMind": ["spar", "AI Safety", "Frontier Safety"],
            "Meta_FAIR": ["PurpleLlama", "Llama Guard"],
        },
        "research_orgs": {
            "Apollo_Research": ["scheming_evals", "alignment_faking"],
            "METR": ["task_evaluation", "autonomy_evals"],
            "MIRI": ["decision_theory", "embedded_agency"],
            "CAIS": ["WMDP", "safety_benchmarks"],
        },
        "policy": {
            "FLI": ["policy_advocacy", "AI_X_risk"],
            "AISC": ["compute_governance", "safety_policies"],
        },
        "debates_open": [
            "interpretability_vs_scalable_oversight",
            "RLHF_vs_constitutional_AI",
            "model_welfare_ethics",
            "open_vs_closed_models",
        ],
    }



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== 28-alignment-research-ecosystem ===")
    print(f"Python {sys.version.split()[0]}")
    funcs = ['alignment_ecosystem_map']
    print(f"Funciones disponibles: {len(funcs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
