# Modelos de vision-lenguaje

> Vision + Language en un solo modelo. Vision encoder (CLIP ViT) + projector (linear o Q-Former) + LLM (LLaMA, Mistral). LLaVA, BLIP-2, GPT-4V, Claude 3.5. La convergencia de vision y NLP.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 18-clip-vocabulario-abierto
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Convertir imagen a tokens.
- Componer image + text tokens.
- Generar respuesta con LLM.
- Diagnosticar LLaVA, BLIP-2, closed vs open.

## Constrúyelo

```python
def llava_compose(image_tokens, text_tokens):
    return np.vstack([image_tokens, text_tokens])
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-vlm-elegir
fase: 04
leccion: 25
---

1. Max accuracy: GPT-4o, Claude 3.5, Gemini 1.5.
2. Self-host: Qwen2-VL, InternVL2, LLaVA-1.6.
3. Custom: LoRA sobre LLM.
4. Mobile: Phi-3.5-vision, LLaVA cuantizada.
5. Entrenar: LLaVA-1.5 recipe.
```

## Ejercicios

1. **Captioning**: implementar pipeline CLIP ViT + LLaMA
   para captioning.
2. **VQA**: implementar VQA con BLIP-2 o LLaVA.
3. **Desafio**: fine-tunear LLaVA-1.6 en 1000 pares
   image-instruction custom con LoRA.

## Lecturas recomendadas

- "Visual Instruction Tuning" (LLaVA, Liu et al., 2023)
- "BLIP-2" (Li et al., 2023)
- "Flamingo" (Alayrac et al., 2022)

---

> 📚 **Adaptación al español** de la lección "[Vision-Language Models]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).