# Multimodal agents computer use

> Multimodal agents computer use: VLM + computer use. Screen understanding (vision encoder), click coordinates (x, y prediction), action prediction (click, type, scroll, back), tool use + planning (ReAct loop: think -> act -> observe -> repeat hasta goal). +SOTA 2024-25. Models: GPT-4V Computer Use (OpenAI 2024 seminal native), Claude Computer Use (Anthropic 2024 +SOTA +reasoning +tool use), OS-Atlas (2024 UI grounding +multilingual), ShowUI (2024 action prediction +efficient), UI-TARS (ByteDance 2024 +SOTA +native +multilingual), Aguvis (2024 +SOTA +pure vision +efficient), Aria-UI, CogAgent. +Insights: -API, +Computer use, +UI grounding, +Reasoning. Frameworks: openai, anthropic, transformers, vLLM, playwright, langchain, browser-use, pyautogui. +Production: GPT-4V + Claude + UI-TARS + OS-Atlas SOTA. +Use cases: web automation, OS control, testing, RPA, UI grounding. Trade-offs: agents + automation, manual + control. Hoy: SOTA 2024-25 standard. 2025: +Native + reasoning + planning + agents + world model.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12/05, 12/06, 12/24
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar encode_screenshot.
- Implementar predict_click_coordinates.
- Implementar predict_action.
- Implementar execute_action y planning_loop (ReAct).
- Implementar os_atlas_ui_grounding y showui_action.
- Diagnosticar GPT-4V vs Claude vs OS-Atlas vs ShowUI vs UI-TARS.

## Constrúyelo

```python
def planning_loop(goal, screenshot, max_steps=5, action_history=None):
    if action_history is None:
        action_history = []
    for step in range(max_steps):
        action = predict_action(screenshot, goal)
        action_history.append(action)
        result = execute_action(action)
        if "done" in str(result).lower():
            break
    return action_history
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
fase: 12
leccion: 25
---

1. VLM + computer use.
2. Click (x, y) prediction.
3. 4 actions.
4. ReAct loop.
5. +SOTA 2024-25.
```

## Ejercicios

1. **GPT-4V Computer Use**:
   usar GPT-4V con openai.
2. **OS-Atlas**: probar
   OS-Atlas UI grounding.
3. **Desafio**: computer
   use agent custom.

## Lecturas recomendadas

- "GPT-4V(ision) System Card" (OpenAI, 2023)
- "Claude Computer Use: An AI Agent That Can Operate Your Computer" (Anthropic, 2024)
- "OS-Atlas: A Foundation Action Model for Generalist GUI Agents" (2024)
- "ShowUI: One Vision-Language-Action Model for GUI Visual Agent" (Lin et al., 2024)

---

> 📚 **Adaptación al español de la lección [Multimodal Agents Computer Use]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).