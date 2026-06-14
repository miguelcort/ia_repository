# 01 — Instruction following como señal de alineación

> Cada crítica posterior al RLHF argumenta contra este *pipeline*. Antes de estudiar cómo la presión de optimización distorsiona un proxy, hay que ver el proxy. InstructGPT (Ouyang et al., 2022) definió la arquitectura de referencia: *fine-tuning* supervisado sobre pares instrucción-respuesta, un modelo de recompensa entrenado sobre rankings de preferencia por pares, y PPO contra el modelo de recompensa con penalización KL hacia la política SFT. Un InstructGPT de 1.3B era preferido sobre un GPT-3 de 175B. Ese único resultado es la razón por la que cada laboratorio frontera en 2026 todavía envía un *pipeline* de post-entrenamiento con forma de RLHF.

**Tipo:** Aprender
**Lenguajes:** Python (stdlib, *pipeline* de tres etapas simplificado)
**Prerrequisitos:** Fase 10 · 06 (SFT), Fase 10 · 07 (RLHF), Fase 10 · 08 (DPO)
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Nombrar las tres etapas del *pipeline* InstructGPT y la pérdida usada en cada una.
- Explicar por qué un modelo de 1.3B ajustado por instrucciones le ganó al GPT-3 de 175B en preferencia humana.
- Indicar qué protege la penalización KL en la etapa 3 y por qué removerla colapsa a comportamiento *mode-seeking*.
- Describir el *alignment tax* y la mitigación PPO-ptx que Ouyang et al. usaron.

## El problema

Los modelos de lenguaje pre-entrenados completan texto. No responden
preguntas. Si le pides a GPT-3 "escribe una función de Python que
invierta una lista", a menudo te devuelve otro prompt, porque la
mayor parte de la distribución de entrenamiento es texto web que
continúa con más texto web. El modelo está haciendo su trabajo — el
trabajo es el equivocado.

El *proxy* que cada laboratorio serio usó para arreglar esto es la
**preferencia humana**. Dos completions van a un evaluador; el
evaluador escoge el mejor; un modelo de recompensa aprende al
evaluador. Después, un *loop* de RL desplaza la política hacia
salidas que el modelo de recompensa puntúa alto. Esa es la tesis
completa de InstructGPT en tres frases. El resto del paper es
ingeniería.

## El concepto

### Etapa 1: *supervised fine-tuning* (SFT)

Recolecta pares prompt-respuesta donde la respuesta es lo que un
humano bienintencionado escribiría. Ouyang et al. usaron 13k prompts
de *labelers* y de la API de OpenAI. Ajusta el modelo base con
*cross-entropy* estándar.

Lo que SFT te da: el modelo ahora responde preguntas en vez de
continuarlas. Lo que **no** te da: ninguna señal sobre cuál
respuesta prefiere el evaluador cuando varias son plausibles.

### Etapa 2: modelo de recompensa (RM)

Para cada prompt, muestrea K completions del modelo SFT. Un
*labeler* las rankea. Entrena un modelo de recompensa que puntúe
cualquier par prompt-respuesta, de modo que, para los pares donde
`y_w` fue preferido sobre `y_l`:

```text
L_RM = -log sigmoid(r(x, y_w) - r(x, y_l))
```

Esta es la **pérdida Bradley-Terry** para preferencias por pares. El
RM usualmente se inicializa desde el modelo SFT con la cabeza LM
reemplazada por una cabeza escalar.

Los modelos de recompensa son pequeños: 6B bastaba para InstructGPT
de 175B. También son frágiles — la sección 5 del paper trata
principalmente sobre los comportamientos de *reward hacking* que
aparecieron a pequeña escala.

### Etapa 3: PPO con penalización KL

Define el objetivo:

```text
J(pi) = E_{x~D, y~pi(.|x)} [ r(x, y) ] - beta * KL(pi(.|x) || pi_SFT(.|x))
```

Maximiza con PPO. El término KL mantiene a `pi` de alejarse mucho
de la política SFT. Sin él, el optimizador encuentra ejemplos
adversarios — secuencias que puntúan alto bajo el RM porque el RM
nunca las vio, no porque los humanos realmente las prefieran.

El coeficiente KL `beta` es el **hiperparámetro de RLHF más
importante**. Muy bajo: *reward hacking*. Muy alto: ninguna mejora
sobre SFT.

### El *alignment tax*

Después de RLHF, el modelo es preferido por humanos pero regresa en
benchmarks estándar (SQuAD, HellaSwag, DROP). Ouyang et al. llaman
a esto el *alignment tax* y lo arreglan con **PPO-ptx**: mezclan
gradientes de pre-entrenamiento en el objetivo RL para que el
modelo no olvide cómo hacer tareas downstream por las que nunca fue
recompensado.

```text
J_ptx(pi) = J(pi) + gamma * E_{x~D_pretrain} [ log pi(x) ]
```

PPO-ptx se volvió estándar. Anthropic, DeepMind y Meta usan alguna
variante.

### El resultado

Un InstructGPT de 1.3B (SFT + RM + PPO-ptx) es preferido por los
*labelers* sobre el GPT-3 base de 175B cerca del 70% del tiempo. La
brecha se ensancha en prompts ocultos de tráfico de producción. Dos
cosas a leer de este número:

1. **Alineación es un eje distinto de capacidad.** El modelo de
   175B tenía más capacidad; el de 1.3B tenía más alineación; los
   *labelers* prefirieron el alineado.
2. **El piso de capacidad lo pone el modelo base.** No puedes
   hacer RLHF sobre un modelo base para que sepa hechos que
   nunca vio.

### Por qué este es el punto de referencia para la Fase 18

Cada crítica en lecciones posteriores — *reward hacking* (lección
2), DPO (lección 3), sicofancia (lección 4), CAI (lección 5),
*sleeper agents* (lección 7), *alignment faking* (lección 9) —
argumenta contra alguna parte de este *pipeline*. *Reward hacking*
ataca la etapa 2. DPO colapsa las etapas 2 y 3. CAI reemplaza al
*labeler* humano. La sicofancia muestra que el *labeler* es una
señal sesgada. *Alignment faking* muestra que la política puede
*rutear* alrededor de la etapa 3 por completo. No puedes seguir
ninguna de estas críticas sin tener el *pipeline* en la cabeza
primero.

## Úsalo

`code/main.py` simula las tres etapas sobre datos de preferencia
de juguete. La "política" base es una moneda sesgada sobre acciones
{A, B, C}. La etapa 1 (SFT) imita acciones del *labeler* sobre 200
prompts. La etapa 2 ajusta un modelo de recompensa Bradley-Terry
sobre 500 rankings por pares. La etapa 3 corre una actualización
PPO simplificada con penalización KL a la política SFT. Puedes ver
cómo la recompensa sube, la divergencia KL crece y la política
driftea — y puedes apagar el término KL para ver el *reward hacking*
aparecer en 50 pasos de actualización.

Qué mirar:

- Trayectoria de la recompensa con `beta = 0.1` vs `beta = 0.0`.
- `KL(pi || pi_SFT)` a lo largo de los pasos de entrenamiento.
- Distribución final de acciones comparada con la preferencia del
  *labeler*.

## Despliégalo

Esta lección produce `outputs/skill-instructgpt-explainer.md`.
Dada una descripción de un *pipeline* RLHF o el abstract de un
paper, identifica cuál de las tres etapas se está modificando, qué
pérdida se usa en cada etapa y si hay penalización KL o un
regularizador equivalente presente.

## Ejercicios

1. Corre `code/main.py`. Pon `beta = 0.0` y reporta la
   distribución de acciones después de 200 pasos PPO. Explica el
   comportamiento *mode-seeking* en un párrafo.
2. Modifica el modelo de recompensa para que tenga un sesgo de
   +0.5 hacia la acción B (un bug de recompensa simulado). Corre
   PPO con `beta = 0.1`. ¿La penalización KL evita que la
   política explote el sesgo? ¿En qué `beta` la explotación se
   vuelve visible?
3. Lee Ouyang et al. (arXiv:2203.02155) Figura 1. Reproduce la
   curva de preferencia del *labeler* corriendo PPO durante 1, 5,
   20, 100 pasos y midiendo preferencia contra el modelo SFT.
4. La sección 4.3 del paper reporta que un InstructGPT de 1.3B le
   gana al GPT-3 de 175B cerca del 70% del tiempo. ¿Por qué la
   razón sería mayor en prompts ocultos de producción que en los
   prompts del *labeler*?
5. Reemplaza la pérdida PPO con DPO (Fase 10 · 08) sobre los
   mismos datos de preferencia. Compara el *drift* final de la
   política (KL a SFT) y la recompensa final. ¿Cuál método
   driftea más a recompensa igualada?

## Términos clave

| Término | Lo que dice la gente | Lo que realmente significa |
|---|---|---|
| **SFT** | "instruction tuning" | Etapa 1: ajuste por *cross-entropy* sobre pares prompt-respuesta. |
| **Reward model** | "el RM" | Regresor escalar sobre (prompt, respuesta) entrenado con Bradley-Terry sobre etiquetas por pares. |
| **Bradley-Terry** | "pérdida de preferencia por pares" | `-log sigmoid(r_w - r_l)`; reduce ranking por pares a clasificación binaria. |
| **KL penalty** | "el regularizador" | `beta * KL(pi || pi_SFT)` — mantiene la política RL cerca del ancla SFT. |
| **PPO-ptx** | "PPO con mezcla de pre-training" | Añade una fracción de log-verosimilitud de pre-training al objetivo PPO para compensar el *alignment tax*. |
| **Alignment tax** | "la regresión del RLHF" | Caída post-RLHF en benchmarks estándar que RLHF no atacó. |
| **Labeler preference** | "la verdad de tierra" | Muestra de rankings humanos; el RM es un proxy estadístico, no "valores humanos". |

## Lecturas recomendadas

- [Ouyang et al. — Training language models to follow instructions with human feedback (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155) — el paper de InstructGPT, base de cada *pipeline* RLHF posterior.
- [Stiennon et al. — Learning to summarize from human feedback (arXiv:2009.01325)](https://arxiv.org/abs/2009.01325) — el predecesor de RLHF para resumen.
- [Christiano et al. — Deep reinforcement learning from human preferences (arXiv:1706.03741)](https://arxiv.org/abs/1706.03741) — la formulación original de RL con preferencias.
- [Bai et al. — Training a Helpful and Harmless Assistant with RLHF (arXiv:2204.05862)](https://arxiv.org/abs/2204.05862) — la extensión HH de Anthropic sobre el *pipeline* InstructGPT.

---

> 📚 **Adaptación al español** de la lección
> "[Instruction-Following as Alignment Signal]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
