# Fase 13 — Herramientas y protocolos

> Las interfaces entre la IA y el mundo real.

Los LLMs por sí solos son **cajas de texto**: leen texto y producen
texto. Para que un LLM sea útil en producción necesita **herramientas**:
funciones externas que el modelo puede invocar (consultar una API,
ejecutar código, leer una base de datos) y **protocolos** que rigen
cómo se describen, descubren y ejecutan esas herramientas. Esta fase
cubre el ecosistema moderno: function calling, MCP (Model Context
Protocol), A2A (Agent-to-Agent), OpenTelemetry para LLMs, gateways
y *skills* de agentes.

La fase se organiza en **cuatro bloques**. El **bloque 1**
(lecciones 1–5) cubre la **interfaz de herramienta**: definición
de tools, function calling, *streaming*, salidas estructuradas y
diseño de schemas. El **bloque 2** (6–14) entra al **Model Context
Protocol (MCP)**: fundamentos, servidores, clientes, transportes,
resources, prompts, sampling, *roots*, *async tasks* y *MCP apps*.
El **bloque 3** (15–18) trata la **seguridad y operación de MCP**:
tool poisoning, OAuth 2.1, gateways/registros y auth en producción.
El **bloque 4** (19–23) cierra con **protocolos de agentes y
observabilidad**: A2A, OpenTelemetry GenAI, routing layer, skills
y SDKs.

## Índice de lecciones

### Bloque 1 — La interfaz de herramienta

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 01 | [La interfaz de herramienta](01-the-tool-interface/) | Aprender | Definición formal: nombre, descripción, schema, efectos. |
| 02 | [Function calling a fondo](02-function-calling-deep-dive/) | Construir | JSON schemas, validación, *parallel tool use*. |
| 03 | [Llamadas paralelas y streaming](03-parallel-and-streaming-tool-calls/) | Construir | Concurrencia, *partial JSON*, SSE. |
| 04 | [Salidas estructuradas](04-structured-output/) | Construir | JSON mode, *grammar-constrained decoding*. |
| 05 | [Diseño de schemas para tools](05-tool-schema-design/) | Aprender | Buenas prácticas, errores comunes, versionado. |

### Bloque 2 — Model Context Protocol (MCP)

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 06 | [Fundamentos de MCP](06-mcp-fundamentals/) | Aprender | Arquitectura, JSON-RPC, capacidades. |
| 07 | [Construir un servidor MCP](07-building-an-mcp-server/) | Construir | Servidor en Python o TypeScript con *tools*. |
| 08 | [Construir un cliente MCP](08-building-an-mcp-client/) | Construir | Cliente que descubre y ejecuta *tools*. |
| 09 | [Transportes MCP](09-mcp-transports/) | Aprender | stdio, HTTP+SSE, StreamableHTTP. |
| 10 | [MCP resources y prompts](10-mcp-resources-and-prompts/) | Construir | Exponer datos y *prompt templates*. |
| 11 | [MCP sampling](11-mcp-sampling/) | Construir | Servidor que pide al cliente llamar al LLM. |
| 12 | [MCP roots y elicitation](12-mcp-roots-and-elicitation/) | Construir | Permisos y solicitudes al usuario. |
| 13 | [MCP async tasks](13-mcp-async-tasks/) | Construir | Tareas de larga duración, *progress notifications*. |
| 14 | [MCP Apps](14-mcp-apps/) | Construir | UI embebida en la respuesta del servidor. |

### Bloque 3 — Seguridad y operación de MCP

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 15 | [Seguridad MCP I: tool poisoning](15-mcp-security-tool-poisoning/) | Aprender | Definiciones maliciosas, *rug pull*, defensa. |
| 16 | [Seguridad MCP II: OAuth 2.1](16-mcp-security-oauth-2-1/) | Construir | PKCE, *authorization servers*, *scopes*. |
| 17 | [Gateways y registros MCP](17-mcp-gateways-and-registries/) | Aprender | Catálogo central, *approval workflows*, auditoría. |
| 18 | [Auth MCP en producción](18-mcp-auth-production/) | Construir | *Token rotation*, *audience binding*, *downscoping*. |

### Bloque 4 — Protocolos de agentes y observabilidad

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 19 | [Protocolo A2A](19-a2a-protocol/) | Construir | *Agent-to-Agent*, *agent cards*, *task delegation*. |
| 20 | [OpenTelemetry GenAI](20-opentelemetry-genai/) | Construir | Semantic conventions, *spans* de tool calls, *eval* spans. |
| 21 | [Routing layer para LLMs](21-llm-routing-layer/) | Aprender | *Smart routing*, *fallback*, *model cascades*. |
| 22 | [Skills y SDKs de agentes](22-skills-and-agent-sdks/) | Aprender | Claude Skills, OpenAI Agents SDK, Google ADK. |
| 23 | [Capstone — ecosistema de herramientas](23-capstone-tool-ecosystem/) | Construir | Servidor MCP + cliente + A2A + OTel. |

## Prerrequisitos

- **Fases 10 y 11** completas.
- Conocimiento de JSON, JSON-RPC y HTTP.
- **Fase 14** (Ingeniería de agentes) — recomendable en paralelo.
- Familiaridad con `asyncio` en Python (o Node.js para servidores
  MCP en TypeScript).

## Objetivos de la fase

Al terminar esta fase el estudiante podrá:

- **Definir** *tools* para LLMs con JSON schemas robustos.
- **Implementar** function calling con OpenAI, Anthropic o Gemini.
- **Construir** un servidor MCP completo en Python o TypeScript.
- **Construir** un cliente MCP que descubra y ejecute herramientas.
- **Asegurar** servidores MCP con OAuth 2.1 y buenas prácticas
  contra tool poisoning.
- **Exponer** *Agent Cards* vía A2A y delegar tareas entre
  agentes.
- **Instrumentar** una app LLM con OpenTelemetry GenAI.
- **Diseñar** un *routing layer* multi-modelo con fallbacks y
  cascades.

## Stack y herramientas

- **OpenAI, Anthropic, Gemini, Mistral** como proveedores.
- **MCP Python SDK** y **MCP TypeScript SDK**.
- **FastMCP** y **MCP Inspector** para desarrollo.
- **LangChain, LlamaIndex** como frameworks (con criterio).
- **OAuth 2.1** y **Keycloak/Auth0** para auth.
- **OpenTelemetry** y **Langfuse/Phoenix/Honeycomb** para
  observabilidad.
- **LiteLLM, Portkey, OpenRouter** para routing.

## Conceptos clave

| Concepto | Aparece en | Reaparece en |
|---|---|---|
| **Function calling** | Lección 02 | Fase 11, 14. |
| **JSON Schema** | Lecciones 02, 05 | Validación. |
| **MCP** | Lecciones 6–18 | Estándar de facto. |
| **JSON-RPC** | Lección 06 | Transporte base de MCP. |
| **A2A** | Lección 19 | Multi-agente. |
| **OpenTelemetry** | Lección 20 | Observabilidad. |
| **Tool poisoning** | Lección 15 | Seguridad. |
| **OAuth 2.1** | Lección 16 | Auth. |

## Cómo estudiar esta fase

1. **Empieza por la lección 01.** Sin entender la interfaz de
   herramienta, MCP parece sobreingeniería.
2. **Construye un servidor MCP propio (lección 07) antes de leer
   sobre seguridad.** Verás por qué la seguridad importa.
3. **La lección 11 (sampling) es la más sorprendente.** Es el
   caso de uso que justifica la complejidad de MCP sobre un
   simple REST.
4. **OpenTelemetry GenAI (lección 20) es transversal.** Aplícala
   a cualquier app LLM que construyas.
5. **El capstone (lección 23) es la prueba final.** Integra
   servidores MCP, A2A y OTel en un solo sistema.

## Verificación de progreso

```bash
# Lección 02 — function calling
python3 fases/13-herramientas-y-protocolos/02-function-calling-deep-dive/code/main.py

# Lección 07 — servidor MCP
python3 fases/13-herramientas-y-protocolos/07-building-an-mcp-server/code/main.py

# Lección 23 — capstone del ecosistema
python3 fases/13-herramientas-y-protocolos/23-capstone-tool-ecosystem/code/main.py
```

Si los tres demos terminan con código 0, la fase está aprobada.

## Cuándo usar cada protocolo

| Caso de uso | Protocolo | Lección |
|---|---|---|
| Tool local simple | Function calling | 02 |
| Tools compartidas entre clientes | MCP | 06–14 |
| Agentes que delegan entre sí | A2A | 19 |
| Observabilidad de cualquier app LLM | OpenTelemetry GenAI | 20 |
| Multi-modelo con fallbacks | Routing layer | 21 |
| Auth en herramientas externas | OAuth 2.1 | 16, 18 |

## Conexión con otras fases

- **Entrada** → [Fase 11 — Ingeniería de LLMs](../11-ingenieria-llms/README.md)
  y [Fase 14 — Ingeniería de agentes](../14-ingenieria-agentes/README.md)
  (idealmente en paralelo).
- **Salida natural** → [Fase 16 — Multi-agente y enjambres](../16-multi-agente-y-enjambres/README.md).
- **Reuso en** → Fase 17 (producción), Fase 19 (capstone).

## Recursos recomendados

- *Model Context Protocol specification* — <https://modelcontextprotocol.io>.
- *A2A protocol specification* — Google, 2025.
- *OpenTelemetry GenAI semantic conventions* — <https://opentelemetry.io/docs/specs/semconv/gen-ai>.
- *OAuth 2.1* — IETF draft.
- *Hugging Face MCP course* — próximamente.

## Véase también

- [glosario/terminos.md](../../glosario/terminos.md) — *function
  calling*, *MCP*, *A2A*, *OAuth 2.1*, *OpenTelemetry*.
- [Fase 11 — Ingeniería de LLMs](../11-ingenieria-llms/README.md).
- [Fase 14 — Ingeniería de agentes](../14-ingenieria-agentes/README.md).
- [ROADMAP.md](../../ROADMAP.md) — estado de las 20 fases.

---

> 📚 **Adaptación al español** del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> de Rohit Ghumare (MIT). Ver [CREDITS.md](../../CREDITS.md).
