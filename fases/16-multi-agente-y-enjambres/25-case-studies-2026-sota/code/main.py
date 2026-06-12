"""
Lección: 25-case-studies-2026-sota
Fase: 16
Case studies 2026 SOTA: latest
multi-agent systems, Devin, AutoGen,
CrewAI, ChatDev, MetaGPT, research
papers, lessons learned.
"""
from __future__ import annotations


CASE_STUDIES = {
    "Devin": {
        "name": "Devin",
        "vendor": "Cognition",
        "year": 2024,
        "type": "autonomous",
        "key_features": ["self-supervised", "long-horizon", "tool use"],
        "results": "resolves 13.86% of SWE-bench issues",
    },
    "AutoGen": {
        "name": "AutoGen",
        "vendor": "Microsoft",
        "year": 2023,
        "type": "framework",
        "key_features": ["conversable agents", "group chat", "code execution"],
        "results": "open source, widely adopted",
    },
    "CrewAI": {
        "name": "CrewAI",
        "vendor": "CrewAI",
        "year": 2023,
        "type": "framework",
        "key_features": ["role-based", "process-driven", "tool integration"],
        "results": "100k+ users",
    },
    "ChatDev": {
        "name": "ChatDev",
        "vendor": "OpenBMB",
        "year": 2023,
        "type": "research",
        "key_features": ["chat chain", "software company sim"],
        "results": "academic paper",
    },
    "MetaGPT": {
        "name": "MetaGPT",
        "vendor": "Geeks",
        "year": 2023,
        "type": "framework",
        "key_features": ["SOPs", "assembly line", "software company"],
        "results": "HumanEval pass@1 85%+",
    },
}


def list_case_studies():
    return list(CASE_STUDIES.keys())


def get_case_study(key):
    return CASE_STUDIES.get(key)


def by_vendor(vendor):
    return [(k, v) for k, v in CASE_STUDIES.items() if v["vendor"].lower() == vendor.lower()]


def by_year(year):
    return [(k, v) for k, v in CASE_STUDIES.items() if v["year"] == year]


def by_type(typ):
    return [(k, v) for k, v in CASE_STUDIES.items() if v["type"] == typ]


def main() -> int:
    print(f"Cases: {list_case_studies()}")
    print(f"Microsoft: {by_vendor('Microsoft')}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())