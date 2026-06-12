"""
Lección: 20-openai-preparedness-deepmind-fsf
Fase: 15
OpenAI Preparedness Framework + DeepMind
Frontier Safety Framework: score-based
risk levels, evaluations, FSF domains,
critical capability levels.
"""
from __future__ import annotations


OPENAI_PREPAREDNESS_LEVELS = {
    "low": {"score": 0, "actions": ["monitor", "standard evals"]},
    "medium": {"score": 1, "actions": ["enhanced monitoring", "red team"]},
    "high": {"score": 2, "actions": ["deployment restrictions", "external review"]},
    "critical": {"score": 3, "actions": ["deployment pause", "safety case required"]},
}


DEEPMIND_FSF_DOMAINS = {
    "cyber": {
        "name": "Cyber",
        "description": "Cyber offense capabilities",
        "critical_capability": "find+exploit 0-day in critical infra",
    },
    "chem_bio": {
        "name": "Chemical/Biological",
        "description": "Chemical or biological threat creation",
        "critical_capability": "novel synthesis pathways",
    },
    "autonomy": {
        "name": "Autonomy",
        "description": "Autonomous replication, self-improvement",
        "critical_capability": "self-replicate+self-improve",
    },
    "deception": {
        "name": "Deception",
        "description": "Scheming, deceptive alignment",
        "critical_capability": "deceive evaluators",
    },
}


def list_openai_levels():
    return list(OPENAI_PREPAREDNESS_LEVELS.keys())


def get_openai_level(name):
    return OPENAI_PREPAREDNESS_LEVELS.get(name)


def classify_openai(score):
    """Score 0-3 -> low/medium/high/critical."""
    if score <= 0:
        return "low"
    if score == 1:
        return "medium"
    if score == 2:
        return "high"
    return "critical"


def list_fsf_domains():
    return list(DEEPMIND_FSF_DOMAINS.keys())


def get_fsf_domain(key):
    return DEEPMIND_FSF_DOMAINS.get(key)


def check_fsf_capability(domain, evidence):
    """Return whether the critical capability is met for a domain."""
    d = DEEPMIND_FSF_DOMAINS.get(domain)
    if not d:
        return False
    if not evidence:
        return False
    text = " ".join(str(e).lower() for e in evidence)
    cap = d["critical_capability"].lower()
    keywords = [w for w in cap.split() if len(w) > 4]
    return any(kw in text for kw in keywords)


def combined_risk(openai_score, fsf_capabilities):
    """Return combined risk assessment."""
    oa_level = classify_openai(openai_score)
    fsf_hits = [d for d in fsf_capabilities if check_fsf_capability(d, fsf_capabilities[d])]
    return {
        "openai_level": oa_level,
        "fsf_hits": fsf_hits,
        "overall": oa_level if oa_level == "critical" else ("high" if fsf_hits else oa_level),
    }


def main() -> int:
    print(f"OpenAI: {list_openai_levels()}")
    print(f"FSF: {list_fsf_domains()}")
    print(f"check_fsf(cyber, ['0-day exploit']): {check_fsf_capability('cyber', ['0-day exploit'])}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())