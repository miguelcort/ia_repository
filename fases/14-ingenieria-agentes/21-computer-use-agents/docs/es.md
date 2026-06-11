# Computer use agents

> Computer use agents: AI control mouse + keyboard + screen. SOTA models: (1) Claude Computer Use (Anthropic 2024 +computer +Anthropic +SOTA), (2) Operator (OpenAI 2024 +browser +cloud +OpenAI), (3) OS-Atlas (2024 +UI +grounding), (4) ShowUI (2024 +efficient +action), (5) UI-TARS (ByteDance 2024 +native +SOTA), (6) Aguvis (2024 +pure vision +SOTA). Loop: (1) observe screenshot, (2) predict action (click/type/scroll), (3) execute (mouse/keyboard +UI), (4) repeat until done. Screen: width + height + elements list (x, y, w, h, type, label), find_element returns center coords. +Computer, +Visual, +UI, +Mouse, +Keyboard, +Screenshots, +Reliable, +Standard, +Production. Variants: Claude, Operator, OS-Atlas, ShowUI, UI-TARS, Aguvis, Aria-UI, CogAgent, custom. Frameworks: anthropic, openai, os-atlas, showui, ui-tars, langchain. +Production: standard 2024-25. +Use cases: computer use, web, OS, automation, UI. Decision: general -> Claude, browser -> Operator, UI -> OS-Atlas, efficient -> ShowUI, native -> UI-TARS, production -> combinacion. Trade-offs: cada uno + specialty. 2025: +MCP + A2A + native + models.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12/25
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar Screen class.
- Implementar ComputerUseAgent.
- Implementar observe + predict + execute.
- Diagnosticar Claude vs Operator vs OS-Atlas.

## Constrúyelo

```python
class ComputerUseAgent:
    def predict_action(self, observation, instruction):
        if "click" in instruction.lower():
            return {"action": "click", "target": instruction.split("click")[-1].strip()}
        return {"action": "done"}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: computer-use
fase: 14
leccion: 21
---

1. Screen.
2. Observe + predict + execute.
3. Click / type / done.
4. Loop.
5. +Computer.
```

## Ejercicios

1. **Computer use**: usar
   Claude Computer Use.
2. **OS-Atlas**: probar
   OS-Atlas con UI grounding.
3. **Desafio**: full
   computer use agent.

## Lecturas recomendadas

- "Claude Computer Use" (Anthropic, 2024)
- "OpenAI Operator" (OpenAI, 2024)
- "OS-Atlas: A Foundation Action Model for Generalist GUI Agents" (2024)
- "ShowUI: One Vision-Language-Action Model for GUI Visual Agent" (Lin et al., 2024)

---

> 📚 **Adaptación al español de la lección [Computer Use Agents]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).