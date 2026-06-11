# Embodied VLAs OpenVLA Pi0 GR00T

> Embodied VLAs (Vision-Language-Action): image + instruction -> robot action (7-DoF: xyz + roll-pitch-yaw + gripper). Discretize (256 bins OpenVLA) o continuous (flow matching Pi0). Models: RT-2 (Google 2023 seminal 55B PaLI-X + action), Octo (UC Berkeley 2024 +open +SOTA small), OpenVLA (2024 +SOTA +open 7B Prismatic VLM), Pi0 (Physical Intelligence 2024 +SOTA flow matching 3B foundation), GR00T (NVIDIA 2024 +foundation humanoid +SOTA), HPT/CrossFormer (Stanford 2024 +transfer +SOTA). Action chunking: predice secuencia de N acciones (10-50) en vez de 1, +temporal coherence -compute -jitter. Frameworks: openvla, lerobot, transformers, roboflow. +Insights: -Single modality, +Action, +Embodied, +Chunking. Production: OpenVLA + Pi0 + GR00T + Octo SOTA. +Use cases: manipulation, navigation, humanoid, mobile, transfer. Trade-offs: VLA + embodied, VLM + chat, VLM video, action chunking + coherence, single + simple. 2025: +Embodied + native + reasoning + humanoid + world model.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12/05, 12/06
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar encode_image_for_vla y encode_instruct.
- Implementar vla_action_decode.
- Implementar vla_forward end-to-end.
- Implementar discretize_actions (256 bins).
- Implementar chunk_actions (10-50 chunking).
- Diagnosticar OpenVLA vs Pi0 vs GR00T.

## Constrúyelo

```python
def vla_forward(image, instruction, action_dim=7, embed_dim=4096):
    img_tokens = encode_image_for_vla(image, embed_dim=embed_dim)
    instr_tokens = encode_instruct(instruction, embed_dim=embed_dim)
    fused = np.concatenate([img_tokens, instr_tokens], axis=0)
    return vla_action_decode(fused.mean(axis=0), action_dim=action_dim)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: vla
fase: 12
leccion: 21
---

1. Vision-Language-Action.
2. 7-DoF actions.
3. Discretize o flow.
4. Chunking 10-50.
5. +SOTA embodied 2024-25.
```

## Ejercicios

1. **OpenVLA**: usar OpenVLA
   con HuggingFace.
2. **Pi0**: probar Pi0
   flow matching.
3. **Desafio**: VLA custom
   para robot.

## Lecturas recomendadas

- "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control" (Brohan et al., 2023)
- "OpenVLA: An Open-Source Vision-Language-Action Model" (Kim et al., 2024)
- "Pi0: A Vision-Language-Action Flow Model for General Robot Control" (Black et al., 2024)
- "GR00T: A Foundation Model for Robotics" (NVIDIA, 2024)

---

> 📚 **Adaptación al español de la lección [Embodied VLAs OpenVLA Pi0 GR00T]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).