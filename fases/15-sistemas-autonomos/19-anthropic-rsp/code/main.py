"""
Lección: 19-anthropic-rsp
Fase: 15
Anthropic Responsible Scaling Policy:
capability levels, deployment thresholds,
safety commitments, evaluation, ASL
standards.
"""
from __future__ import annotations


RSP_LEVELS = {
    "ASL-2": {
        "name": "AI Safety Level 2",
        "description": "Current frontier models; standard safety practices",
        "compute_threshold_pflops": 1e25,
        "requirements": ["harmlessness training", "basic evals", "incident reporting"],
        "year": 2023,
    },
    "ASL-3": {
        "name": "AI Safety Level 3",
        "description": "Models with significantly elevated capabilities",
        "compute_threshold_pflops": 1e26,
        "requirements": [
            "red team evaluation",
            "safety cases",
            "responsible scaling officer",
            "autonomous replication evals",
        ],
        "year": 2024,
    },
    "ASL-4": {
        "name": "AI Safety Level 4",
        "description": "Models with substantial autonomous capabilities",
        "compute_threshold_pflops": 1e28,
        "requirements": [
            "frontier safety evaluations",
            "alignment audits",
            "external oversight",
            "deployment restrictions",
        ],
        "year": 2025,
    },
}


def list_levels():
    return list(RSP_LEVELS.keys())


def get_level(name):
    return RSP_LEVELS.get(name)


def classify_model(compute_pflops, capability_indicators=None):
    """Return the RSP level for a given compute budget."""
    capability_indicators = capability_indicators or []
    levels_sorted = sorted(
        RSP_LEVELS.items(),
        key=lambda kv: kv[1]["compute_threshold_pflops"],
    )
    current = "ASL-2"
    for level, info in levels_sorted:
        if compute_pflops >= info["compute_threshold_pflops"]:
            current = level
    if capability_indicators:
        if any(i in capability_indicators for i in ["autonomous_replication", "cyber_offense"]):
            if current == "ASL-2":
                current = "ASL-3"
        if "deceptive_alignment" in capability_indicators:
            current = "ASL-4"
    return current


def check_safety_case(model_name, safety_evidence):
    """Verify safety case requirements for a given level."""
    required_keys = ["harmlessness_eval", "alignment_eval", "deployment_plan"]
    missing = [k for k in required_keys if k not in safety_evidence]
    return {
        "model": model_name,
        "complete": len(missing) == 0,
        "missing": missing,
        "evidence_count": len(safety_evidence),
    }


def main() -> int:
    print(f"Levels: {list_levels()}")
    print(f"classify(1e27): {classify_model(1e27)}")
    print(f"classify(1e29, ['deceptive_alignment']): {classify_model(1e29, ['deceptive_alignment'])}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())