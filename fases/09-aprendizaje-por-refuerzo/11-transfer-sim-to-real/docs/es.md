# Transfer sim-to-real

> Sim-to-real: entrenar policy en simulación, deploy en real. Challenges: reality gap, long-tail, safety. Soluciones: Domain Randomization (visual, dynamics, sensor), System ID, domain adaptation (CycleGAN), real-world fine-tuning. Aplicaciones: robotics (OpenAI Rubik's cube), drones, autonomous driving, LLM agents. Frameworks: Isaac Sim, MuJoCo, PyBullet, Isaac Gym. Frontier: foundation models + sim-to-real.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 09/05-dqn, 09/08-ppo
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar domain randomization.
- Aplicar observation noise.
- Calcular system ID residuals.
- Diagnosticar mitigations del reality gap.

## Constrúyelo

```python
def domain_randomization_param(name, distribution, low, high, seed=0):
    rng = np.random.default_rng(seed)
    if distribution == "uniform":
        return rng.uniform(low, high)
    elif distribution == "normal":
        return rng.normal((low + high) / 2, (high - low) / 4)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-sim-to-real
fase: 09
leccion: 11
---

1. DR randomize params.
2. SysID ajustar sim.
3. Visual, dynamics, sensor DR.
4. Reality gap mitigations.
5. LLM agents sim-to-real.
```

## Ejercicios

1. **DR**: entrenar policy con DR
   en MuJoCo CartPole.
2. **SysID**: ajustar friction y mass
   con data real.
3. **Desafio**: sim-to-real para
   drone racing.

## Lecturas recomendadas

- "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World" (Tobin et al., 2017)
- "Solving Rubik's Cube with a Robot Hand" (OpenAI, 2019)
- "Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics" (Zhao et al., 2020)

---

> 📚 **Adaptación al español** de la lección "[Sim to Real Transfer]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).