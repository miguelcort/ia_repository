# 17 — Personal AI tutor

> Personal AI tutor (Khanmigo, Khan Academy, Photomath): adaptive learning, Socratic method, misconceptions tracking, progress. Sistemas: long-term memory, knowledge tracing, multimodal (text + diagram + voice).

**Tipo:** Capstone
**Lenguajes:** Python
**Prerrequisitos:** Fase 11, Fase 12 (multimodal), Fase 14
**Tiempo estimado:** 25 horas

## Objetivos

- Knowledge tracing.
- Socratic questioning.
- Misconception detection.
- Eval sobre learning outcomes.

## El problema

AI tutors (Khanmigo, Photomath, Socratic by Google)
combinan: (1) Knowledge tracing (Bayesian
Knowledge Tracing, DKT). (2) Socratic method
(preguntas en vez de respuestas directas). (3)
Misconception detection. (4) Adaptive difficulty.
(5) Long-term memory (estudiante profile). (6)
Multimodal: texto, diagramas, voz. Métricas:
learning gain, time to mastery, retention.
Cuidado: dependencia del estudiante, equity, calidad
de explicación.

## Constrúyelo

```python
class PersonalTutor:
    def __init__(self, llm, kt_model):
        self.llm = llm
        self.kt = kt_model
        self.student_profile = {}

    def teach(self, topic, question):
        mastery = self.kt.estimate(topic, self.student_profile)
        if mastery < 0.7:
            return self.socratic_question(topic, question)
        else:
            return self.advance_topic(topic, question)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-personal-tutor
fase: 19
leccion: 17
---

1. Knowledge tracing.
2. Socratic questioning.
3. Misconception detection.
4. Long-term memory.
5. Learning gain eval.
```

## Ejercicios

1. **BKT**: implementar
   Bayesian Knowledge Tracing.
2. **Socratic**: 50 preguntas
   en math.
3. **Desafío**: learning
   gain eval.

## Lecturas recomendadas

- "Khanmigo" (Khan Academy 2024)
- "Deep Knowledge Tracing" (Piech 2015)
- "Socratic AI Tutors" (Markel 2023)
- "Photomath" (2023)

---

> 📚 **Adaptación al español** de la lección
> "[17-personal-ai-tutor]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
