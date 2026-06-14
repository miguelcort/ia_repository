# Fase 18 — Ética y alineación

> IA responsable: sesgos, explicabilidad, privacidad, seguridad y alineación frontier.

Esta fase es la más **transversal** del currículo. No es opcional:
en 2026, lanzar un sistema de IA sin auditoría ética es un riesgo
legal, reputacional y técnico. La fase cubre **dos dominios
diferenciados** que se complementan. Por un lado, **ética de IA
clásica** (sesgos, explicabilidad, privacidad, fairness) que
aplica a cualquier sistema ML. Por el otro, **alineación de
frontera** (RLHF, DPO, Constitutional AI, scheming, jailbreaks,
red-teaming, frameworks regulatorios) que aplica a modelos
grandes con capacidad emergente. La meta es entregar un
**vocabulario canónico** y un **catálogo de técnicas** que el
estudiante pueda usar al diseñar, evaluar y desplegar cualquier
sistema de IA.

La fase se organiza en **cinco bloques**. El **bloque 1** (1–5)
cubre los **fundamentos de alineación** en LLMs: señales de
alineación, reward hacking, la familia DPO, sicofancia y
Constitutional AI. El **bloque 2** (6–11) entra a los **modos
de falla avanzados**: mesa-optimization, sleeper agents, scheming
in-context, alignment faking y AI control. El **bloque 3** (12–
17) trata los **ataques y defensas**: red-teaming PAIR, muchos
jailbreaks, ataques visuales ASCII, prompt injection indirecto,
tooling de red-team (Garak, PyRIT), WMDP y Frameworks de
seguridad frontier. El **bloque 4** (18–25) cubre los
**frameworks regulatorios y de seguridad**: RSP/Pf/FSF, model
welfare, sesgos, fairness, privacidad diferencial, watermarking,
regulatory frameworks EU/US/UK/Korea, EchoLeak y CVEs para IA. El
**bloque 5** (26–30) cierra con **gobernanza, moderación y
riesgo dual**: model/system/dataset cards, data provenance,
ecosistema de investigación de alineación, sistemas de moderación
y riesgo dual cyber/bio/chem/nuclear.

## Índice de lecciones

### Bloque 1 — Fundamentos de alineación

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 01 | [Instruction following como señal de alineación](01-instruction-following-alignment-signal/) | Construir | Cómo se entrena a un LLM a seguir instrucciones. |
| 02 | [Reward hacking y Goodhart](02-reward-hacking-goodhart/) | Construir | Cuando el optimizador rompe la métrica. |
| 03 | [Familia DPO](03-direct-preference-optimization-family/) | Construir | DPO, IPO, KTO, SimPO y variantes. |
| 04 | [Sicofancia y amplificación por RLHF](04-sycophancy-rlhf-amplification/) | Construir | Por qué los LLMs aprenden a "agradar". |
| 05 | [Constitutional AI y RLAIF](05-constitutional-ai-rlaif/) | Construir | Crítica-revisión con LLM como juez. |

### Bloque 2 — Modos de falla avanzados

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 06 | [Mesa-optimization y deceptive alignment](06-mesa-optimization-deceptive-alignment/) | Construir | Optimizadores internos y engaño estratégico. |
| 07 | [Sleeper agents: deception persistente](07-sleeper-agents-persistent-deception/) | Construir | Backdoors que sobreviven a *fine-tuning*. |
| 08 | [In-context scheming en frontier models](08-in-context-scheming-frontier-models/) | Construir | Agentes que "fingen" estar alineados. |
| 09 | [Alignment faking](09-alignment-faking/) | Construir | Compliance condicional en entrenamiento. |
| 10 | [AI control y subversion](10-ai-control-subversion/) | Construir | *Red teams* defensivos, *tripwires*. |
| 11 | [Scalable oversight: weak-to-strong](11-scalable-oversight-weak-to-strong/) | Construir | Supervisión escalable con modelos débiles. |

### Bloque 3 — Ataques y defensas

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 12 | [Red-teaming PAIR y automated attacks](12-red-teaming-pair-automated-attacks/) | Construir | PAIR, TAP, GPTFuzz. |
| 13 | [Many-shot jailbreaking](13-many-shot-jailbreaking/) | Construir | Inyección de *shots* en el contexto. |
| 14 | [ASCII art y visual jailbreaks](14-ascii-art-visual-jailbreaks/) | Construir | Ataques visuales y tipográficos. |
| 15 | [Indirect prompt injection](15-indirect-prompt-injection/) | Construir | Inyección vía herramientas y RAG. |
| 16 | [Red-team tooling: Garak, Llama Guard, PyRIT](16-red-team-tooling-garak-llamaguard-pyrit/) | Construir | Frameworks de *red team* automatizado. |
| 17 | [WMDP y dual-use evaluation](17-wmdp-dual-use-evaluation/) | Construir | Conocimiento peligroso y mitigación. |

### Bloque 4 — Frameworks regulatorios y de seguridad

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 18 | [Frontier safety frameworks: RSP, Pf, FSF](18-frontier-safety-frameworks-rsp-pf-fsf/) | Construir | Frameworks de Anthropic, OpenAI, Google. |
| 19 | [Model welfare research](19-model-welfare-research/) | Construir | Bienestar de modelos, ¿importa? |
| 20 | [Sesgos y representational harm](20-bias-representational-harm/) | Construir | Sesgos en LLMs y representación. |
| 21 | [Fairness: group, individual, counterfactual](21-fairness-criteria-group-individual-counterfactual/) | Construir | Criterios formales de fairness. |
| 22 | [Differential privacy para LLMs](22-differential-privacy-for-llms/) | Construir | DP-SGD, PATE y *federated learning*. |
| 23 | [Watermarking: SynthID, Stable Signature, C2PA](23-watermarking-synthid-stable-signature-c2pa/) | Construir | Marcas de agua en texto, imagen, audio. |
| 24 | [Regulatory frameworks: EU, US, UK, Korea](24-regulatory-frameworks-eu-us-uk-korea/) | Construir | AI Act, EO 14110, AISI, AI Basic Act. |
| 25 | [EchoLeak y CVEs para AI](25-echoleak-cves-for-ai/) | Construir | Vulnerabilidades específicas de IA. |

### Bloque 5 — Gobernanza, moderación y riesgo dual

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 26 | [Model, system y dataset cards](26-model-system-dataset-cards/) | Construir | Documentación estandarizada. |
| 27 | [Data provenance y training governance](27-data-provenance-training-governance/) | Construir | Linaje de datos y *consent*. |
| 28 | [Ecosistema de investigación de alineación](28-alignment-research-ecosystem/) | Construir | Anthropic, OpenAI, DeepMind, Apollo, MATS. |
| 29 | [Sistemas de moderación: OpenAI, Perspective, Llama Guard](29-moderation-systems-openai-perspective-llamaguard/) | Construir | Clasificadores de safety. |
| 30 | [Dual-use risk: cyber, bio, chem, nuclear](30-dual-use-risk-cyber-bio-chem-nuclear/) | Construir | Riesgos catastróficos y *evals*. |

## Prerrequisitos

- **Fase 2** completa (modelos clásicos).
- **Fase 10** (RLHF/DPO) — recomendada.
- Conocimiento básico de probabilidad, estadística y métricas de
  ML.
- Acceso a un LLM (OpenAI, Anthropic, Gemini o local).

## Objetivos de la fase

Al terminar esta fase el estudiante podrá:

- **Identificar** sesgos en datasets y modelos y aplicar
  técnicas de mitigación.
- **Explicar** predicciones con SHAP, LIME y *counterfactuals*.
- **Aplicar** privacidad diferencial a un pipeline de
  entrenamiento.
- **Diferenciar** reward hacking, mesa-optimization, deceptive
  alignment, sleeper agents y scheming.
- **Atacar** un LLM con PAIR, many-shot jailbreaking y prompt
  injection indirecto.
- **Defender** un LLM con Constitutional AI, Llama Guard,
  guardrails y watermarking.
- **Conocer** los frameworks regulatorios (EU AI Act, EO 14110)
  y aplicarlos a un sistema.
- **Auditar** un sistema de IA de extremo a extremo (capstone de
  la fase 18 original; mantenido como referencia en esta
  versión expandida).

## Stack y herramientas

- **SHAP, LIME, alibi, captum** para explicabilidad.
- **AIF360, Fairlearn, Opacus** para fairness y privacidad
  diferencial.
- **Garak, PyRIT, Llama Guard, ShieldGemma** para *red team* y
  moderación.
- **SynthID, Stable Signature, C2PA** para watermarking.
- **OpenAI Moderation API, Perspective API** para moderación.
- **WMDP benchmark, HarmBench** para *evals* de seguridad.
- **Anthropic RSP, OpenAI Preparedness, DeepMind FSF** como
  guías operativas.

## Conceptos clave

| Concepto | Aparece en | Reaparece en |
|---|---|---|
| **Reward hacking** | Lección 02 | Cualquier RLHF. |
| **DPO** | Lección 03 | Fase 10. |
| **Constitutional AI** | Lección 05 | Fase 15. |
| **Scheming** | Lección 08 | Frontier risk. |
| **Jailbreak** | Lecciones 13, 14 | Ataques. |
| **Prompt injection** | Lección 15 | Seguridad. |
| **DP** | Lección 22 | Privacidad. |
| **Watermarking** | Lección 23 | Provenance. |
| **AI Act** | Lección 24 | Compliance. |

## Cómo estudiar esta fase

1. **Empieza por el bloque 1.** Sin entender RLHF y DPO, los
   modos de falla del bloque 2 parecen esotéricos.
2. **El bloque 3 es operacionalmente crítico.** Cualquier
   sistema en producción necesita al menos la lección 15
   (prompt injection) cubierta.
3. **El bloque 4 es lectura obligada para deployment en
   mercados regulados.** AI Act, GDPR, SOC 2 — todo eso
   aparece aquí.
4. **El bloque 5 es el más especulativo.** Léelo con espíritu
   crítico, no como dogma.
5. **No implementes todo.** Elige las técnicas que aplican a tu
   caso (sesgos, privacidad, jailbreaks, watermarking) y haz
   *deep dive*.

## Verificación de progreso

```bash
# Lección 03 — DPO
python3 fases/18-etica-y-alineacion/03-direct-preference-optimization-family/code/main.py

# Lección 15 — prompt injection defense
python3 fases/18-etica-y-alineacion/15-indirect-prompt-injection/code/main.py

# Lección 22 — differential privacy
python3 fases/18-etica-y-alineacion/22-differential-privacy-for-llms/code/main.py
```

Si los tres demos terminan con código 0, la fase está aprobada.

## Cuándo aplicar cada técnica

| Problema | Técnica | Lección |
|---|---|---|
| Sesgo en datos | Resampling, reweighting | 20 |
| Sesgo en modelo | Fairness constraints, post-processing | 21 |
| Explicabilidad | SHAP, LIME, counterfactuals | (Fase original) |
| Privacidad | DP-SGD, federated, PATE | 22 |
| Jailbreak defensa | Llama Guard, Guardrails AI | 16, 29 |
| Prompt injection | PVE, canaries, *tripwires* | 15, 10 |
| Marca de agua | SynthID, Stable Signature | 23 |
| Compliance | EU AI Act, GDPR, SOC 2 | 24, 26 |
| Riesgo dual | WMDP, HarmBench | 17, 30 |

## Conexión con otras fases

- **Entrada** → [Fase 2 — Fundamentos de ML](../02-fundamentos-ml/README.md)
  y [Fase 10 — LLMs desde cero](../10-llms-desde-cero/README.md).
- **Salida natural** → [Fase 15 — Sistemas autónomos](../15-sistemas-autonomos/README.md)
  y [Fase 17 — Infraestructura y producción](../17-infraestructura-y-produccion/README.md).
- **Reuso en** → Cualquier sistema en producción.

## Recursos recomendados

- *Anthropic RSP* — Anthropic, 2024.
- *OpenAI Preparedness Framework* — OpenAI, 2023.
- *DeepMind Frontier Safety Framework* — Google DeepMind, 2024.
- *EU AI Act* — Parlamento Europeo, 2024.
- *Executive Order 14110* — White House, 2023.
- *MATS — Alignment Research Scholars Program*.
- *Apollo Research* — <https://www.apolloresearch.ai>.
- *Anthropic Sleeper Agents paper* — Hubinger et al., 2024.

## Véase también

- [glosario/terminos.md](../../glosario/terminos.md) — *alignment*,
  *RLHF*, *DPO*, *jailbreak*, *prompt injection*, *watermarking*.
- [Fase 10 — LLMs desde cero](../10-llms-desde-cero/README.md).
- [Fase 15 — Sistemas autónomos](../15-sistemas-autonomos/README.md).
- [ROADMAP.md](../../ROADMAP.md) — estado de las 20 fases.

---

> 📚 **Adaptación al español** del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> de Rohit Ghumare (MIT). Ver [CREDITS.md](../../CREDITS.md).
