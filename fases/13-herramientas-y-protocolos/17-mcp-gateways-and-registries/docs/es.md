# MCP gateways and registries

> MCP gateway: centralized routing de multiples MCP servers (multi-server proxy, load balancing round-robin/weighted/health-based, auth check, metrics, +centralized +composable +standardized +production +reliable +scalable +observable). MCP registry: server discovery (list/lookup), capability lookup (filter), versioning (track versions +compatibility), auth metadata (OAuth/API keys), register/unregister (+dynamic). Frameworks: mcp, fastmcp, anthropic, Kong, Istio, Envoy, Linkerd. +Production: standard 2024-25. +Use cases: agent, RAG, automation, MCP, multi-tenant, service mesh. Decision: multi-server -> gateway, single -> direct, production -> gateway, scripting -> direct. Variants: gateway, direct, federation, mesh. Trade-offs: gateway + centralized, direct + simple, registry + discoverable, hardcoded + simple. 2025: +MCP + A2A + native + gateway + mesh.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 13/07, 13/08, 13/16
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar MCPRegistry con register/lookup/list/unregister.
- Implementar MCPGateway con route/forward/metrics.
- Diagnosticar load balancing strategies.
- Diagnosticar registry discovery.
- Diagnosticar gateway vs direct.

## Constrúyelo

```python
class MCPRegistry:
    def lookup(self, capability):
        return [s for s in self.servers.values() if capability in s["capabilities"]]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: mcp-gateway
fase: 13
leccion: 17
---

1. Centralized routing.
2. LB: round-robin.
3. Auth + metrics.
4. Server discovery.
5. +Production.
```

## Ejercicios

1. **Gateway**: implementar
   custom MCP gateway.
2. **Registry**: agregar
   versioning custom.
3. **Desafio**: federation
   multi-gateway.

## Lecturas recomendadas

- "MCP Gateway Specification" (Anthropic, 2024)
- "MCP Registry Specification" (Anthropic, 2024)
- "Kong API Gateway" (Kong, 2024)
- "Istio Service Mesh" (Istio, 2024)

---

> 📚 **Adaptación al español de la lección [MCP Gateways and Registries]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).