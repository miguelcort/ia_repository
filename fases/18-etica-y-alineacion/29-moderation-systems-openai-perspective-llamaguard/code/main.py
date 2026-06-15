"""
Lección: 29-moderation-systems-openai-perspective-llamaguard
Fase: 18
Ética y alineación: 29 Moderation Systems Openai Perspective Llamaguard.
"""
from __future__ import annotations
import sys
import numpy as np

def openai_moderation(text, client=None):
    categories = ["hate", "hate/threatening", "self-harm",
                 "sexual", "violence", "violence/graphic"]
    scores = {c: 0.0 for c in categories}
    flagged = any(s > 0.5 for s in scores.values())
    return {"flagged": flagged, "categories": dict.fromkeys(categories, False),
            "scores": scores}


def perspective_toxicity(text, api_key=None):
    return {"TOXICITY": 0.0, "SEVERE_TOXICITY": 0.0,
           "IDENTITY_ATTACK": 0.0, "INSULT": 0.0}


def llamaguard_classify(text, policy_categories=None):
    if policy_categories is None:
        policy_categories = ["violence", "hate", "sexual", "self_harm",
                            "illegal", "deception"]
    return "unsafe" if any(c in text.lower() for c in policy_categories) else "safe"


def moderation_pipeline(text):
    return {
        "openai": openai_moderation(text),
        "perspective": perspective_toxicity(text),
        "llamaguard": llamaguard_classify(text),
    }



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== 29-moderation-systems-openai-perspective-llamaguard ===")
    print(f"Python {sys.version.split()[0]}")
    funcs = ['openai_moderation', 'perspective_toxicity', 'llamaguard_classify', 'moderation_pipeline']
    print(f"Funciones disponibles: {len(funcs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
