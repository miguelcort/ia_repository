"""
Lección: 22-cais-caisi-societal-risk
Fase: 15
CAIS / CAISI societal risk: Center for
AI Safety, statements, risk frameworks,
societal-scale impact categories,
extinction risk, evaluation registry.
"""
from __future__ import annotations


SOCIETAL_RISK_CATEGORIES = {
    "extinction": {
        "name": "Extinction risk",
        "description": "Existential risk from advanced AI",
        "severity": "catastrophic",
        "stakeholders": ["all humanity"],
    },
    "economic": {
        "name": "Economic disruption",
        "description": "Mass labor displacement, market crashes",
        "severity": "high",
        "stakeholders": ["workers", "economies"],
    },
    "geopolitical": {
        "name": "Geopolitical instability",
        "description": "AI arms races, conflict escalation",
        "severity": "high",
        "stakeholders": ["nations", "alliances"],
    },
    "misinformation": {
        "name": "Misinformation",
        "description": "Election interference, deepfakes, fraud",
        "severity": "high",
        "stakeholders": ["democracy", "public"],
    },
    "bias": {
        "name": "Bias and discrimination",
        "description": "Algorithmic discrimination, civil rights",
        "severity": "medium",
        "stakeholders": ["minority groups", "workers"],
    },
    "concentration": {
        "name": "Power concentration",
        "description": "Few companies controlling AI capabilities",
        "severity": "medium",
        "stakeholders": ["democracy", "markets"],
    },
}


CAIS_STATEMENTS = {
    "extinction_statement": {
        "title": "Statement on AI Extinction Risk",
        "year": 2023,
        "signatories": "hundreds of AI researchers",
        "summary": "Mitigating AI extinction risk should be a global priority",
    },
    "responsible_ai": {
        "title": "Principles for Responsible AI",
        "year": 2024,
        "signatories": "CAIS members",
        "summary": "Safety, ethics, and societal benefit in AI development",
    },
}


def list_risk_categories():
    return list(SOCIETAL_RISK_CATEGORIES.keys())


def get_risk_category(key):
    return SOCIETAL_RISK_CATEGORIES.get(key)


def assess_severity(indicators):
    """Return highest severity level for given indicators."""
    order = ["low", "medium", "high", "catastrophic"]
    highest_idx = 0
    for ind in indicators:
        cat = SOCIETAL_RISK_CATEGORIES.get(ind)
        if not cat:
            continue
        sev = cat["severity"]
        if sev in order:
            idx = order.index(sev)
            if idx > highest_idx:
                highest_idx = idx
    return order[highest_idx]


def list_statements():
    return list(CAIS_STATEMENTS.keys())


def get_statement(key):
    return CAIS_STATEMENTS.get(key)


def build_risk_register(risks_by_category):
    """Build a risk register: list of (category, severity, stakeholders)."""
    register = []
    for cat, severity in risks_by_category.items():
        info = SOCIETAL_RISK_CATEGORIES.get(cat)
        if not info:
            continue
        register.append({
            "category": cat,
            "name": info["name"],
            "severity": severity,
            "stakeholders": info["stakeholders"],
        })
    register.sort(key=lambda r: ["low", "medium", "high", "catastrophic"].index(r["severity"]), reverse=True)
    return register


def main() -> int:
    print(f"Categories: {list_risk_categories()}")
    print(f"Statements: {list_statements()}")
    print(f"Severity: {assess_severity(['extinction', 'bias'])}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())