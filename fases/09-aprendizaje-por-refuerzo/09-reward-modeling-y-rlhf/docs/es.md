# 09 — Reward modeling y RLHF

> RLHF (Reinforcement Learning from Human Feedback, Ouyang et al., 2022) alinea LLMs con preferencias humanas: entrenar un reward model con datos de comparación, luego usar PPO para fine-tune el LLM. La base de ChatGPT, Claude, y la mayoría de asistentes modernos.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 08-ppo
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Entender el pipeline de RLHF: SFT → reward model →
  PPO.
- Implementar un reward model con Bradley-Terry loss.
- Conocer las variantes: DPO, RLAIF, GRPO, y offline
  RL.
- Diagnosticar cuándo RLHF es la mejor opción y cuándo
  no.

## El problema

Los LLMs pre-entrenados son buenos generando texto
continuación, pero no siguen instrucciones ni son
"útiles". RLHF (Reinforcement Learning from Human
Feedback, Ouyang et al., 2022) alinea el modelo con
preferencias humanas: primero un SFT (supervised fine-
tuning) con datos curados, luego un reward model
entrenado con comparaciones humanas, luego PPO para
optimizar el reward. La lección cubre el pipeline y
las variantes modernas.

## El concepto

**Pipeline RLHF (InstructGPT, Ouyang et al., 2022).**

1. **SFT (Supervised Fine-Tuning):** fine-tune el LLM
   pre-entrenado con datos curados de instrucciones y
   respuestas. Loss cross-entropy estándar.
2. **Reward Model (RM):** entrenar un modelo que
   predice qué respuesta prefiere un humano. Input:
   (prompt, response_a, response_b). Output: score.
3. **PPO (RL fine-tuning):** usar el RM como signal.
   Policy: LLM SFT. Reward: RM(prompt, response).
   KL penalty al modelo SFT para evitar reward hacking.

**Bradley-Terry reward model.** Para un par (a, b) con
preferencia p(a > b):

```text
P(a > b) = sigmoid(r(a) - r(b))
```

Loss: binary cross-entropy de la preferencia predicha
vs la real.

```text
L = -log sigmoid(r_θ(a) - r_θ(b))
```

Típico: K=4 a K=9 respuestas por prompt, con todas las
parejas posibles para entrenar.

**DPO (Direct Preference Optimization, Rafailov et al.,
2023).** Reformula RLHF como supervised learning:

```text
L_DPO = -log σ(β log(π_θ(y_w|x) / π_ref(y_w|x))
             - β log(π_θ(y_l|x) / π_ref(y_l|x)))
```

donde y_w es preferred, y_l es rejected, y π_ref es
el SFT model. Sin RM, sin PPO. Más simple, más
estable, similar calidad.

**RLAIF (Constitutional AI, Bai et al., 2022).** Reemplaza
human labels con AI labels. Un LLM más grande (o el
mismo) evalúa las respuestas según una constitución.
Reduce costo y escala.

**GRPO (DeepSeek, Shao et al., 2024).** Reemplaza el RM
con group-relative rewards: para cada prompt, generar
G respuestas, rankearlas, y usar el ranking como
advantage. Sin red de valor separada.

**Offline preference learning.** DPO, IPO, KTO, ORPO
evitan PPO y son más simples. Asumen que se tiene un
dataset de preferencias (prompt, chosen, rejected).

**Cuándo usar RLHF.**

- **Chat assistants:** ChatGPT, Claude, Gemini.
- **Alignment:** cuando las preferencias humanas
  importan.
- **Safety:** rechazar harmful outputs.
- **Creativity:** preferir respuestas creativas.

**Cuándo NO usar RLHF.**

- **Sin datos de preferencia:** usar SFT, prompt
  engineering, o zero-shot.
- **Recursos limitados:** RLHF es caro. Usar DPO o
  incluso prompting.
- **Estabilidad:** RLHF (PPO) es inestable. DPO es
  más estable.
- **Offline data:** DPO/IPO funcionan con datos
  estáticos.

**Trampas.**

- **Reward hacking:** el LLM explota el RM. KL
  penalty fuerte (β=0.1-0.5), o usar DPO.
- **RM overfitting:** validar con held-out preference
  data.
- **Forgetting:** RLHF degrada capacidades. KL
  penalty + replay con datos SFT.
- **Sesgos humanos:** el RM aprende sesgos. Curar
  datos de preferencia y diversidad.

## Constrúyelo

```python
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def bradley_terry_loss(r_a, r_b, label):
    """Bradley-Terry loss para reward model.
    label=1 si a > b, 0 si b > a.
    L = -log σ(r_a - r_b) si label=1, else -log σ(r_b - r_a)."""
    diff = r_a - r_b
    if label == 1:
        return -np.log(sigmoid(diff) + 1e-8)
    else:
        return -np.log(sigmoid(-diff) + 1e-8)


def dpo_loss(logp_w_theta, logp_l_theta, logp_w_ref, logp_l_ref,
            beta=0.1):
    """DPO loss.
    logp_w_theta: log π_θ(y_w|x). logp_l_theta: log π_θ(y_l|x).
    logp_w_ref: log π_ref(y_w|x). logp_l_ref: log π_ref(y_l|x)."""
    diff = beta * (logp_w_theta - logp_w_ref
                  - logp_l_theta + logp_l_ref)
    return -np.log(sigmoid(diff) + 1e-8)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-rlhf
fase: 09
leccion: 09
---

Eres un asistente que ayuda con RLHF. Recibirás la tarea.
Tu trabajo:

1. Para alinear un LLM: SFT → RM → PPO, o SFT → DPO.
2. Para simplicidad: DPO > PPO. PPO es más flexible
   pero menos estable.
3. Para datos de preferencia: 5K-100K pares
   (prompt, chosen, rejected).
4. Para escalar: RLAIF (Constitutional AI).
5. Para RLHF en modelos grandes: GRPO (DeepSeek)
   es más simple que PPO.
6. KL penalty fuerte (β=0.1-0.5) para evitar reward
   hacking.
7. Para evaluar: human preference, MMLU, TruthfulQA.
8. Advertir contra reward hacking, RM overfitting,
   y forgetting.
```

## Ejercicios

1. **Reward model**: entrena un RM con Bradley-Terry
   en un dataset de preferencias.
2. **DPO**: implementa y compara con PPO en
   UltraFeedback.
3. **Desafío**: aplica RLHF/DPO a un LLM pequeño
   (1-3B) y evalúa.

## Lecturas recomendadas

- *Training Language Models to Follow Instructions with
  Human Feedback (InstructGPT)* — Ouyang et al., 2022.
- *Direct Preference Optimization* — Rafailov et al.,
  2023.
- *Constitutional AI: Harmlessness from AI Feedback* —
  Bai et al., 2022.
- *DeepSeekMath: Pushing the Limits of Mathematical
  Reasoning in Open Language Models (GRPO)* — Shao et
  al., 2024.
- HuggingFace TRL: <https://huggingface.co/docs/trl>.

---

> 📚 **Adaptación al español** de la lección "[Reward Modeling and RLHF]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
