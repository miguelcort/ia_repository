"""
Lección: 27-data-provenance-training-governance
Fase: 18
Ética y alineación: 27 Data Provenance Training Governance.
"""
from __future__ import annotations
import sys
import numpy as np

def data_provenance_record(source_url, license, crawl_date,
                         transformation="", opt_out_status="ok"):
    return {
        "source": source_url,
        "license": license,
        "crawled": crawl_date,
        "transformation": transformation,
        "opt_out": opt_out_status,
        "checksum": hash(source_url + str(crawl_date)),
    }


def copyright_filter(dataset, blocked_sources):
    return [d for d in dataset
           if d.get("source") not in blocked_sources]


def opt_out_check(dataset, opt_out_registry):
    return [d for d in dataset
           if d.get("source") not in opt_out_registry]


def training_governance_policy(data_point):
    license = data_point.get("license", "")
    if license in ["public_domain", "cc-by", "cc0", "cc-by-sa"]:
        return "allow"
    if license in ["all-rights-reserved", "unknown", ""]:
        return "review"
    return "block"



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== 27-data-provenance-training-governance ===")
    print(f"Python {sys.version.split()[0]}")
    funcs = ['data_provenance_record', 'copyright_filter', 'opt_out_check', 'training_governance_policy']
    print(f"Funciones disponibles: {len(funcs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
