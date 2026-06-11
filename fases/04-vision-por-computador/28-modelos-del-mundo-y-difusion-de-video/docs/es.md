# Modelos del mundo y difusión de video

> La frontera de vision generativa: world models que aprenden la dinamica del entorno (RL, robotics) y video diffusion que produce segundos de video coherente. Sora, Wan2.1, DreamerV3 marcan el estado del arte.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 23-difusion-transformers-y-flujo-rectificado
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar spacetime tubelet embedding.
- Implementar positional encoding 3D.
- Implementar un world model step basico.
- Diagnosticar DreamerV3 vs Sora vs Wan2.1.

## Constrúyelo

```python
def spacetime_patches(video, patch_t=2, patch_h=8, patch_w=8):
    T, H, W, C = video.shape
    n_t, n_h, n_w = T // patch_t, H // patch_h, W // patch_w
    patches = video.reshape(n_t, patch_t, n_h, patch_h, n_w, patch_w, C)
    patches = patches.transpose(0, 2, 4, 1, 3, 5, 6)
    return patches.reshape(n_t * n_h * n_w, -1)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-world-video
fase: 04
leccion: 28
---

1. World model RL: DreamerV3, STORM.
2. Video max calidad: Sora, Veo 2, Wan2.1.
3. Video open: Wan2.1, HunyuanVideo, CogVideoX.
4. Autonomus: GAIA-1, DriveDreamer.
5. DiT spacetime + flow matching.
```

## Ejercicios

1. **Video DiT**: implementar DiT con tubelet embedding
   en vez de patch.
2. **Dreamer simple**: RSSM con VAE + RNN, entrena con
   secuencias.
3. **Desafio**: fine-tunear Wan2.1 con LoRA en 20
   videos custom de 2 segundos.

## Lecturas recomendadas

- "DreamerV3" (Hafner et al., 2024)
- "Video Diffusion Models" (Ho et al., 2022)
- "Sora" technical report (OpenAI, 2024)

---

> 📚 **Adaptación al español** de la lección "[World Models and Video Diffusion]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).