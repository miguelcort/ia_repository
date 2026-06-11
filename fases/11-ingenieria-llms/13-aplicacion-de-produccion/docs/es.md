# Aplicación de producción

> Production LLM apps: FastAPI + async + streaming (SSE, WebSocket, gRPC), error handling (retry con exponential backoff 1s/2s/4s/8s, circuit breaker, fallbacks, +tools: tenacity, aiohttp), rate limiting (token bucket, per-user, per-IP), auth (API keys, OAuth, JWT, secrets manager), observability (logs, metrics Prometheus, traces OpenTelemetry, eval RAGAS+LM-judge, alerts), CI/CD (GitHub Actions, GitLab CI), testing (unit, integration, eval), error codes (429 rate limit, 500 server, 503 overloaded, 401 auth, 408 timeout, 400 context length). Frameworks: LangServe (LangChain), FastAPI, BentoML, Vercel AI SDK, Modal, Replicate. Observability SOTA: LangSmith, LangFuse, Helicone, Portkey, OpenLLMetry, Datadog. SOTA 2024-25: full stack (FastAPI + streaming + retry + fallbacks + observability + safety + caching + scaling).

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11/06-rag, 11/09-function-calling
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar streaming chunk.
- Implementar retry con exponential backoff.
- Implementar rate limit check.
- Diagnosticar error responses.
- Diagnosticar frameworks SOTA.

## Constrúyelo

```python
def retry_with_backoff(max_retries=3, base_delay=1):
    return [base_delay * (2 ** i) for i in range(max_retries)]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: production-app
fase: 11
leccion: 13
---

1. FastAPI + async + streaming.
2. Retry + backoff + circuit breaker.
3. Rate limiting + auth.
4. Observability.
5. LangServe, BentoML.
```

## Ejercicios

1. **Streaming**: implementar
   FastAPI + SSE.
2. **Retry**: implementar
   tenacity retry.
3. **Desafio**: deploy
   completo con observability.

## Lecturas recomendadas

- "Building LLM Apps: A Developer's Guide" (Boucher, 2024)
- "LangServe Documentation" (LangChain, 2024)
- "OpenTelemetry for LLM Apps" (OpenLLMetry, 2024)
- "Designing Data-Intensive Applications" (Kleppmann, 2017)

---

> 📚 **Adaptación al español** de la lección "[Production App]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).