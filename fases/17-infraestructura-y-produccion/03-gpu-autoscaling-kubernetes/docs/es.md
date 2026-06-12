# GPU autoscaling Kubernetes

> GPU autoscaling: (1) HPA (metrics+CPU/GPU), (2) KEDA (event-driven+queues), (3) Karpenter (bin-packing+just-in-time), (4) Custom (queue depth+latency), (5) Node pools (GPU types+spot/OD). NodePool: name+gpu_type+count+hourly_cost+available() count-allocated+allocate(n) check. Cluster: pools list+total_gpus() sum+available_gpus() sum+allocate(n) iterate+scale_up(pool, delta). decide_scale: thresholds (up > 70, down < 30, stable else). Diferencias: (1) HPA = metrics+simple+CPU/GPU, (2) KEDA = event+queues+scale to zero, (3) Karpenter = bin-packing+JIT+cost. Criterios: Spot = cost+fault-tolerant+preemptible, On-demand = stable+pay per use+reliable, Reserved = predict+commit+stable. Decision: burst -> spot, critical -> on-demand, steady -> reserved, mix -> spot+OD. Frameworks: kubernetes, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + autoscaling.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17/02
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar NodePool con GPU type + count + cost.
- Implementar Cluster con pools + total + available.
- Implementar allocate + scale_up.
- Implementar decide_scale con thresholds.
- Diagnosticar HPA vs KEDA vs Karpenter.

## Constrúyelo

```python
class Cluster:
    def allocate(self, n=1):
        for p in self.pools:
            if p.allocate(n):
                return p.name
        return None
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: gpu-autoscaling-kubernetes
fase: 17
leccion: 03
---

1. NodePool + Cluster.
2. allocate + scale_up.
3. decide_scale.
4. +Production.
```

## Ejercicios

1. **NodePool**: probar
   allocate.
2. **Cluster**: probar
   scale_up.
3. **Desafio**: integrar
   con Karpenter.

## Lecturas recomendadas

- "Kubernetes HPA" (Kubernetes, 2024)
- "KEDA" (KEDA, 2024)
- "Karpenter" (AWS, 2024)

---

> 📚 **Adaptación al español de la lección [GPU Autoscaling Kubernetes]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).