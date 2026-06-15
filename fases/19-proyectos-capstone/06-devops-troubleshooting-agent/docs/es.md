# 06 — DevOps troubleshooting agent

> DevOps troubleshooting agent: lee logs, identifica issues, aplica fixes via SSH/API. Sistemas como Shoreline, Resolve, Blink. Multi-source telemetry (logs, metrics, traces), runbook lookup, automated remediation.

**Tipo:** Capstone
**Lenguajes:** Python
**Prerrequisitos:** Fase 14 (agents), Fase 17 (infra), Fase 11
**Tiempo estimado:** 25 horas

## Objetivos

- Implementar agent con log analysis.
- Diagnóstico multi-source (logs, metrics, traces).
- Runbook lookup y execution.
- Evaluar MTTR (mean time to resolve).

## El problema

DevOps agents automatizan troubleshooting: ingieren
logs (Loki, CloudWatch), metrics (Prometheus),
traces (Tempo, Jaeger), y diagnostican issues.
Stack típico: (1) Log parser (LLM + regex).
(2) Anomaly detection (PromQL + ML). (3) Runbook
search (RAG sobre docs internos). (4) Remediation
(SSH, kubectl, AWS API). (5) Validation. (6) Post-
mortem generator. Mide MTTR vs on-call humano.

## Constrúyelo

```python
class DevOpsAgent:
    def __init__(self, log_client, metric_client, runbook_rag):
        self.logs = log_client
        self.metrics = metric_client
        self.runbooks = runbook_rag

    def troubleshoot(self, alert):
        # 1. Gather context
        logs = self.logs.query(alert.service, alert.window)
        metrics = self.metrics.query(alert.service, alert.window)
        # 2. Diagnose
        diagnosis = self.llm.diagnose(logs, metrics, alert)
        # 3. Find runbook
        runbook = self.runbooks.search(diagnosis)
        # 4. Apply fix
        result = self.apply_fix(runbook)
        return {"diagnosis": diagnosis, "fix": runbook,
                "result": result}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-devops-agent
fase: 19
leccion: 06
---

1. Multi-source ingest (logs, metrics, traces).
2. LLM diagnosis.
3. Runbook RAG.
4. Safe remediation.
5. MTTR eval.
```

## Ejercicios

1. **Alert flow**: integrar
   PagerDuty API.
2. **Runbook RAG**: indexar
   100 runbooks.
3. **Desafío**: safe-mode
   con human approval.

## Lecturas recomendadas

- "Shoreline.io" (2024)
- "Blink Ops" (2024)
- "Resolve AI" (2024)
- "Datadog Bits AI" (2024)

---

> 📚 **Adaptación al español** de la lección
> "[06-devops-troubleshooting-agent]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
