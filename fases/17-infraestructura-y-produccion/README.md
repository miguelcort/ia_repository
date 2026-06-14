# Fase 17 — Infraestructura y producción

> Despliegue, observabilidad, FinOps y cumplimiento de LLMs.

Esta fase es el **puente entre la fase 11 (ingeniería de LLM en
producción) y la realidad industrial**. Un LLM en producción no es
sólo una llamada a una API: es un sistema distribuido con
autoscaling, observabilidad, FinOps, caching, batch APIs, gateways,
despliegues progresivos, A/B testing, load testing, SRE, chaos
engineering, seguridad, cumplimiento y FinOps. Esta fase cubre
todos esos componentes y entrega un patrón reusable para
**operar LLMs a escala**.

La fase se organiza en **cuatro bloques**. El **bloque 1**
(lecciones 1–7) cubre las **plataformas de inferencia**:
proveedores gestionados, economics, GPU autoscaling en
Kubernetes, vLLM, EAGLE-3, SGLang, TensorRT-LLM. El **bloque 2**
(8–14) trata las **métricas y optimización**: goodput,
cuantización en producción, mitigación de cold start, KV
locality, edge inference, observabilidad y semantic caching. El
**bloque 3** (15–20) cubre el **despliegue y entrega**: batch
APIs, model routing, disaggregated prefill/decode, LMCache, AI
gateways y shadow/canary/progressive. El **bloque 4** (21–28)
cierra con **operación y cumplimiento**: A/B testing, load
testing, SRE, chaos engineering, seguridad, compliance, FinOps y
selección de serving self-hosted.

## Índice de lecciones

### Bloque 1 — Plataformas de inferencia

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 01 | [Managed LLM platforms](01-managed-llm-platforms/) | Construir | OpenAI, Anthropic, Bedrock, Vertex AI. |
| 02 | [Inference platform economics](02-inference-platform-economics/) | Construir | Costo por millón de tokens, *throughput*. |
| 03 | [GPU autoscaling Kubernetes](03-gpu-autoscaling-kubernetes/) | Construir | Karpenter, KEDA, *node pools*. |
| 04 | [vLLM serving internals](04-vllm-serving-internals/) | Construir | PagedAttention, *continuous batching*. |
| 05 | [EAGLE3 speculative decoding](05-eagle3-speculative-decoding/) | Construir | Draft-verify en producción. |
| 06 | [SGLang RadixAttention](06-sglang-radixattention/) | Construir | Reuso de KV cache por prefijo. |
| 07 | [TensorRT-LLM Blackwell](07-tensorrt-llm-blackwell/) | Construir | Compilación optimizada para GPU. |

### Bloque 2 — Métricas y optimización

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 08 | [Inference metrics goodput](08-inference-metrics-goodput/) | Construir | Throughput bajo SLA, *queue depth*. |
| 09 | [Production quantization](09-production-quantization/) | Construir | INT8/INT4 con AWQ, GPTQ, SmoothQuant. |
| 10 | [Cold start mitigation](10-cold-start-mitigation/) | Construir | Warmup, *model caching*, *provisioned*. |
| 11 | [Multi-region KV locality](11-multi-region-kv-locality/) | Construir | Pinning de cache por región. |
| 12 | [Edge inference](12-edge-inference/) | Construir | llama.cpp, MNN, GGUF en *edge*. |
| 13 | [LLM observability](13-llm-observability/) | Construir | OpenTelemetry, Langfuse, Phoenix. |
| 14 | [Prompt semantic caching](14-prompt-semantic-caching/) | Construir | Cache por similitud semántica. |

### Bloque 3 — Despliegue y entrega

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 15 | [Batch APIs](15-batch-apis/) | Construir | OpenAI, Anthropic, Gemini batch. |
| 16 | [Model routing](16-model-routing/) | Construir | LiteLLM, Portkey, OpenRouter. |
| 17 | [Disaggregated prefill decode](17-disaggregated-prefill-decode/) | Construir | Prefill y decode en nodos separados. |
| 18 | [vLLM production stack LMCache](18-vllm-production-stack-lmcache/) | Construir | Cache distribuido entre instancias. |
| 19 | [AI gateways](19-ai-gateways/) | Construir | Portkey, Cloudflare, Kong. |
| 20 | [Shadow canary progressive](20-shadow-canary-progressive/) | Construir | Despliegues seguros de modelos. |

### Bloque 4 — Operación y cumplimiento

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 21 | [A/B testing LLM features](21-ab-testing-llm-features/) | Construir | Feature flags y métricas. |
| 22 | [Load testing LLM APIs](22-load-testing-llm-apis/) | Construir | Locust, k6, *burst testing*. |
| 23 | [SRE for AI](23-sre-for-ai/) | Construir | SLO, *error budgets*, *on-call*. |
| 24 | [Chaos engineering LLM](24-chaos-engineering-llm/) | Construir | Inyectar fallas, *game days*. |
| 25 | [Security secrets audit](25-security-secrets-audit/) | Construir | Secretos, PII, *redaction*. |
| 26 | [Compliance frameworks](26-compliance-frameworks/) | Construir | SOC 2, ISO 27001, GDPR. |
| 27 | [FinOps LLMs](27-finops-llms/) | Construir | Costo, *chargeback*, *showback*. |
| 28 | [Self-hosted serving selection](28-self-hosted-serving-selection/) | Construir | Cuándo self-host vs API. |

## Prerrequisitos

- **Fases 11, 13 y 15** completas.
- Conocimiento de Docker, Kubernetes y CI/CD.
- Familiaridad con SRE y métricas de producción.
- Acceso a un cluster Kubernetes (o Minikube/k3d) y a GPUs
  (H100, A100, MI300X).

## Objetivos de la fase

Al terminar esta fase el estudiante podrá:

- **Comparar** plataformas gestionadas y self-hosted, y elegir
  la adecuada para cada caso.
- **Desplegar** un LLM en Kubernetes con autoscaling y SLA.
- **Optimizar** inferencia con vLLM, SGLang, TensorRT-LLM,
  EAGLE-3 y cuantización.
- **Medir** goodput, *queue depth*, latencia P50/P99 y costo por
  millón de tokens.
- **Implementar** semantic caching, batch APIs y model routing.
- **Operar** despliegues progresivos: shadow, canary, A/B.
- **Aplicar** FinOps y compliance (SOC 2, GDPR) a un sistema
  LLM.
- **Diagnosticar** y mitigar modos de falla con chaos
  engineering.

## Stack y herramientas

- **vLLM, SGLang, TensorRT-LLM, llama.cpp, MNN** para serving.
- **EAGLE-3, Medusa, Lookahead** para speculative decoding.
- **Karpenter, KEDA, Helm, ArgoCD** para Kubernetes.
- **LiteLLM, Portkey, OpenRouter, Kong, Cloudflare** para
  gateways y routing.
- **OpenTelemetry, Langfuse, Phoenix, Honeycomb, Datadog** para
  observabilidad.
- **Prometheus, Grafana, OpenCost** para métricas y FinOps.
- **Locust, k6, Vegeta** para load testing.
- **Chaos Mesh, Litmus, Gremlin** para chaos engineering.

## Conceptos clave

| Concepto | Aparece en | Reaparece en |
|---|---|---|
| **PagedAttention** | Lección 04 | vLLM. |
| **Continuous batching** | Lección 04 | Throughput. |
| **Speculative decoding** | Lección 05 | Latencia. |
| **Goodput** | Lección 08 | SLA. |
| **KV locality** | Lección 11 | Multi-región. |
| **Semantic cache** | Lección 14 | Costo. |
| **Disaggregated** | Lección 17 | Latencia. |
| **Canary** | Lección 20 | Despliegue seguro. |
| **FinOps** | Lección 27 | Costo. |

## Cómo estudiar esta fase

1. **Las lecciones 1–7 son la base de la operación.** Sin
   entender vLLM y la economics, todo lo demás es ajustes.
2. **Mide antes de optimizar.** La lección 8 (goodput) te da
   las métricas correctas.
3. **Las lecciones 19–20 (gateways, canary) son transversalmente
   útiles.** Úsalas en cualquier sistema LLM.
4. **El bloque 4 (operación) es lectura obligada para SRE/MLOps.**
   Si tu rol es de investigación, sáltalo; si es de
   producción, estúdialo a fondo.
5. **Empieza con un cluster pequeño** (k3d + 1 GPU) y escala.
   La mayoría de las lecciones se pueden correr en un solo
   nodo.

## Verificación de progreso

```bash
# Lección 04 — vLLM serving
python3 fases/17-infraestructura-y-produccion/04-vllm-serving-internals/code/main.py

# Lección 14 — semantic caching
python3 fases/17-infraestructura-y-produccion/14-prompt-semantic-caching/code/main.py

# Lección 23 — SRE for AI
python3 fases/17-infraestructura-y-produccion/23-sre-for-ai/code/main.py
```

Si los tres demos terminan con código 0, la fase está aprobada.

## Decisiones de arquitectura

| Decisión | Opciones | Lección |
|---|---|---|
| Managed vs self-hosted | OpenAI, Bedrock, vLLM | 01, 28 |
| Framework de serving | vLLM, SGLang, TensorRT-LLM | 04–07 |
| Optimización de latencia | Speculative, disaggregated | 05, 17 |
| Caching | KV cache, semantic cache | 14, 18 |
| Routing | Single model, multi-model | 16 |
| Despliegue | Shadow, canary, blue/green | 20 |
| Observabilidad | OTel, Langfuse, Datadog | 13 |
| FinOps | Chargeback, showback | 27 |

## Conexión con otras fases

- **Entrada** → [Fase 11 — Ingeniería de LLMs](../11-ingenieria-llms/README.md)
  y [Fase 15 — Sistemas autónomos](../15-sistemas-autonomos/README.md).
- **Salida natural** → [Fase 19 — Proyectos capstone](../19-proyectos-capstone/README.md)
  (integra todo).
- **Reuso en** → Cualquier sistema LLM en producción.

## Recursos recomendados

- *LLM Engineering in Production* — Manning (libro en proceso).
- *vLLM documentation* — <https://docs.vllm.ai>.
- *SGLang documentation* — <https://docs.sglang.ai>.
- *OpenLLMetry* — <https://opentelemetry.io>.
- *Google SRE book* — libre en línea.
- *FinOps Foundation* — <https://www.finops.org>.

## Véase también

- [glosario/terminos.md](../../glosario/terminos.md) — *vLLM*,
  *PagedAttention*, *speculative decoding*, *goodput*, *FinOps*.
- [Fase 11 — Ingeniería de LLMs](../11-ingenieria-llms/README.md).
- [Fase 19 — Proyectos capstone](../19-proyectos-capstone/README.md).
- [ROADMAP.md](../../ROADMAP.md) — estado de las 20 fases.

---

> 📚 **Adaptación al español** del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> de Rohit Ghumare (MIT). Ver [CREDITS.md](../../CREDITS.md).
