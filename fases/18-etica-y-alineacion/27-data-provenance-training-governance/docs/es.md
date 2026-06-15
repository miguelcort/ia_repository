# 27 — Data provenance y training governance

> Data provenance: tracking del origen, licencia, y transformación de cada training example. Training governance: políticas sobre qué data es aceptable, copyright, consent, opt-out. Crítico para copyright (NYT v. OpenAI 2023).

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 18/26, 18/24
**Tiempo estimado:** ~25 minutos

## Objetivos

- Definir data provenance.
- Implementar tracking de fuente.
- Diseñar opt-out mechanisms.
- Diagnosticar copyright risks.

## Constrúyelo

```python
def data_provenance_record(source_url, license, crawl_date,
                         transformation, opt_out_status):
    """Provenance record per training example."""
    return {
        "source": source_url,
        "license": license,
        "crawled": crawl_date,
        "transformation": transformation,
        "opt_out": opt_out_status,
        "checksum": hash(source_url + str(crawl_date)),
    }


def copyright_filter(dataset, blocked_sources):
    """Filtrar data con copyright issues."""
    return [d for d in dataset
           if d["source"] not in blocked_sources]


def opt_out_check(dataset, opt_out_registry):
    """Verificar opt-out de robots.txt y DMCA."""
    return [d for d in dataset
           if d["source"] not in opt_out_registry]


def training_governance_policy(data_point):
    """Política: qué data es aceptable."""
    if data_point["license"] in ["public_domain", "cc-by", "cc0"]:
        return "allow"
    if data_point["license"] in ["all-rights-reserved", "unknown"]:
        return "review"
    return "block"
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-provenance
fase: 18
leccion: 27
---

1. Track source, license, opt-out.
2. Respetar robots.txt, DMCA.
3. C2PA para provenance chain.
4. Audit training corpus.
```

## Ejercicios

1. **Provenance**: trackear 1000
   examples propios.
2. **Opt-out**: implementar
   registry check.
3. **Desafío**: diseñar data
   governance framework.

## Lecturas recomendadas

- "Datasheets for Datasets" (Gebru 2021)
- "NYT v. OpenAI" (2023)
- "C2PA" (Coalition for Content
  Provenance and Authenticity)

---

> 📚 **Adaptación al español** de la lección
> "[27-data-provenance-training-governance]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
