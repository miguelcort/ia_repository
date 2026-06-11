# ControlNet y LoRA condicionamiento

> LoRA (Hu 2021): low-rank adaptation W_new = W + (α/r)·A·B, A random init, B=0, trainable params <<<< total. ControlNet (Zhang 2023): zero conv + copia del U-Net, conditioning espacial (edges, depth, pose, seg). QLoRA: 4-bit base + LoRA, 65B en 48GB GPU. IP-Adapter: image prompt via CLIP. Hoy: ControlNet + LoRA + IP-Adapter son estándar en SD/SDXL/SD3/FLUX.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 08/07-difusion-latente-stable-diffusion
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar LoRA forward y merge.
- Implementar zero convolution init.
- Diagnosticar ControlNet conditioning.
- Aplicar QLoRA para LLM fine-tune.

## Constrúyelo

```python
def lora_init(A_shape, B_shape, rank=4, seed=0):
    A = rng.standard_normal((A_shape, rank))
    B = np.zeros((rank, B_shape))
    return A, B

def merge_lora(W, A, B, alpha=1.0):
    return W + (alpha / rank) * (A @ B)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-controlnet-lora
fase: 08
leccion: 08
---

1. LoRA: rank 4-128, alpha/r.
2. ControlNet: zero conv, spatial cond.
3. QLoRA: 4-bit + LoRA, paged optim.
4. Conditioning: text, image, spatial.
5. IP-Adapter: CLIP image emb.
```

## Ejercicios

1. **LoRA SD**: entrenar LoRA custom en
   30 imagenes con kohya_ss.
2. **ControlNet**: aplicar depth2image en
   custom dataset.
3. **Desafio**: QLoRA fine-tune Llama 3 8B
   en tu propio dataset.

## Lecturas recomendadas

- "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- "Adding Conditional Control to Text-to-Image Diffusion Models" (Zhang et al., 2023)
- "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
- "IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models" (Ye et al., 2023)

---

> 📚 **Adaptación al español** de la lección "[ControlNet LoRA Conditioning]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).