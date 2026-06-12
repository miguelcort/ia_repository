# Agent economies

> Agent economies: (1) Budget (allocation+caps), (2) Token markets (buy/sell+rates), (3) Bidding (auctions+resources), (4) Spending caps (max+period), (5) Agent-as-customer (API+meter). Budget: total+remaining+spend(amount, who) check+subtract+reset() restore+spent() diff. ResourceMarket: bids list+resources dict+list_resource(name, qty, reserve)+submit_bid+run_auction (sort by amount+top wins). allocate_budget: greedy (sort by cost+by remaining). Ventajas agent economies: distributed (no central+local), incentive (per-agent+aligned), efficient (price signal+market), fair (bid+pay), adaptive (dynamic+responsive). Criterios: Auction = scarce+demand+variable, Fixed = stable+predictable+cheap, Caps = hard limit+period+safe. Decision: scarce -> auction, stable -> fixed, limit -> caps, mix -> caps+auction. Frameworks: langchain, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + economies.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 16/20
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar Budget con spend + reset + spent.
- Implementar ResourceMarket con submit_bid + run_auction.
- Implementar allocate_budget greedy.
- Diagnosticar economies vs central planning.
- Diagnosticar auction vs fixed vs caps.

## Constrúyelo

```python
class Budget:
    def spend(self, amount, who="agent"):
        if amount > self.remaining:
            return False, "exceeded"
        self.remaining -= amount
        self.history.append({"amount": amount, "who": who})
        return True, "ok"
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: agent-economies
fase: 16
leccion: 21
---

1. Budget + spend.
2. ResourceMarket + auction.
3. allocate_budget.
4. +Production.
```

## Ejercicios

1. **Budget**: probar
   spend + reset.
2. **Market**: probar
   auction.
3. **Desafio**: integrar
   con API rate limiter.

## Lecturas recomendadas

- "Multi-Agent Markets" (Wellman, 1996)
- "Mechanism Design" (Mas-Colell, 1995)
- "Cloud Economics" (Weinman, 2012)

---

> 📚 **Adaptación al español de la lección [Agent Economies]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).