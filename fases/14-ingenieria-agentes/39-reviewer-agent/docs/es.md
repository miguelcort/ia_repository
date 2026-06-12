# Reviewer agent

> Reviewer agent: (1) Separate agent (independent+specialized), (2) Reviews output (post-hoc+quality), (3) Checklist-based (items+weights), (4) Structured feedback (PASS/FAIL+score), (5) Approval/rejection (threshold+decision). ReviewChecklist: items list+add(name, fn, weight) Dict+weight+list() Names. Reviewer: checklist+threshold, review(output) iterate+passed/failed+weighted score+approved+feedback. Score: sum weights (total), sum passed weights (passed_weight), score = passed_weight/total (0 if 0), approved = score >= threshold. Feedback: PASS lines+FAIL lines+joined \n. Criterios: Reviewer = post-hoc+single+checklist, Gates = pre-deploy+fast+automated, Multi-agent debate = complex+costly+diverse. Decision: post-hoc -> reviewer, pre-deploy -> gates, complex -> debate, mix -> all. Frameworks: langchain, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + review.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/38
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar ReviewChecklist con items + add + list.
- Implementar Reviewer con threshold + review + score.
- Implementar feedback con PASS/FAIL lines.
- Diagnosticar weighted score.
- Diagnosticar reviewer vs other.

## Constrúyelo

```python
class Reviewer:
    def review(self, output):
        passed = []
        failed = []
        for item in self.checklist.items:
            try:
                ok = bool(item["fn"](output))
                if ok:
                    passed.append(item["name"])
                else:
                    failed.append(item["name"])
            except Exception as e:
                failed.append((item["name"], str(e)))
        total_weight = sum(i["weight"] for i in self.checklist.items)
        passed_weight = sum(i["weight"] for i in self.checklist.items if i["name"] in passed)
        score = passed_weight / total_weight if total_weight > 0 else 0
        approved = score >= self.threshold
        return {"approved": approved, "score": score, "passed": passed, "failed": failed}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: reviewer-agent
fase: 14
leccion: 39
---

1. ReviewChecklist.
2. Reviewer + threshold.
3. weighted score.
4. +Production.
```

## Ejercicios

1. **Reviewer**: probar
   threshold + score.
2. **Feedback**: probar
   PASS/FAIL.
3. **Desafio**: integrar
   con multi-agent debate.

## Lecturas recomendadas

- "Anthropic: Constitutional AI" (Bai et al., 2022)
- "Reviewer Agents" (LangChain, 2024)
- "Self-Consistency" (Wang et al., 2022)

---

> 📚 **Adaptación al español de la lección [Reviewer Agent]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).