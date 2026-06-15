"""
Lección: 26-model-system-dataset-cards
Fase: 18
Ética y alineación: 26 Model System Dataset Cards.
"""
from __future__ import annotations
import sys
import numpy as np

def model_card(model, training_data, evals_results, intended_use,
             out_of_scope, ethical_considerations):
    return {
        "model_details": {
            "name": model.get("name", ""),
            "version": model.get("version", ""),
            "type": model.get("architecture", ""),
            "parameters": model.get("n_parameters", 0),
        },
        "intended_use": intended_use,
        "training_data": {
            "datasets": training_data,
        },
        "evaluations": evals_results,
        "limitations": model.get("known_limitations", []),
        "out_of_scope": out_of_scope,
        "ethical_considerations": ethical_considerations,
    }


def dataset_card(dataset):
    return {
        "dataset_name": dataset.get("name", ""),
        "description": dataset.get("description", ""),
        "language": dataset.get("languages", []),
        "size": dataset.get("size", 0),
        "splits": dataset.get("splits", {}),
        "considerations": {
            "biases": dataset.get("known_biases", []),
            "limitations": dataset.get("limitations", []),
        },
    }


def system_card(agent_system, safety_evaluations):
    return {
        "system": agent_system.get("description", ""),
        "deployment": agent_system.get("deployment_context", ""),
        "safety": safety_evaluations,
        "external_reviewers": agent_system.get("reviewers", []),
    }



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== 26-model-system-dataset-cards ===")
    print(f"Python {sys.version.split()[0]}")
    funcs = ['model_card', 'dataset_card', 'system_card']
    print(f"Funciones disponibles: {len(funcs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
