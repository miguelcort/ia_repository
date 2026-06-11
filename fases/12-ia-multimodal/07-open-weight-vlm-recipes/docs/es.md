# Open weight VLM recipes

> Open VLM recipes SOTA: LLaVA-1.5 (Vicuna-13B + CLIP-ViT-L-336px, MLP projection, 558K instruct, seminal), LLaVA-Next (34B, any resolution 1-4 tiles, +SOTA), Idefics2 (Mistral-7B + SigLIP, Perceiver Resampler, +multilingual, +efficient), OpenFlamingo (MPT/RedPajama + CLIP, gated cross-attn, +few-shot), Molmo (OLMo-7B + CLIP, point ref + OCR, PixMo dataset 712K images / 2.7M points, +SOTA 12+ benchmarks), InternVL3 (4B-72B, +multilingual, +SOTA), Qwen-VL (2B-72B, +multilingual, +dynamic FPS). Training stages: (1) Pretrain projector (LCS-558K), (2) Multimodal pretrain, (3) Visual instruction tuning (LLaVA-Instruct 558K GPT-4V generated), (4) RLHF (LLaVA-RLHF 10K), (5) DPO/ORPO. +Datasets: LCS-558K, LLaVA-Instruct 558K, LLaVA-Next 1.4M, LLaVA-RLHF 10K, PixMo. Frameworks: transformers, vLLM, SGLang, open_clip, trl, peft. Hoy: SOTA 2024-25 es InternVL3 + Qwen-VL + Molmo + LLaVA-Next. +Trade-offs: cada uno tiene specialty (Multilingual, OCR, pointing, any res).

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12/05, 12/06
**Tiempo estimado:** ~30 minutos

## Objetivos

- Listar VLM open weight recipes.
- Filtrar por parameter count.
- Mapear training stages a datasets.
- Diagnosticar PixMo.
- Diagnosticar SOTA 2024-25.

## Constrúyelo

```python
def recipes_by_param_count(min_b=7, max_b=70):
    result = []
    for r in VLM_RECIPES.values():
        b = r["params"] / 1e9
        if min_b <= b <= max_b:
            result.append(r)
    return result
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: open-vlm-recipes
fase: 12
leccion: 07
---

1. LLaVA, LLaVA-Next, Idefics2.
2. OpenFlamingo, Molmo.
3. InternVL3, Qwen-VL.
4. 4+ training stages.
5. PixMo + OCR.
```

## Ejercicios

1. **LLaVA**: entrenar LLaVA
   con llava repository.
2. **Molmo**: usar Molmo
   con pointing task.
3. **Desafio**: comparar
   VLM en benchmark.

## Lecturas recomendadas

- "Visual Instruction Tuning" (Liu et al., 2023)
- "Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models" (Deitke et al., 2024)
- "Idefics2: Efficient and Open Multimodal Model" (Laurençon et al., 2024)
- "OpenFlamingo: An Open-Source Framework for Training Few-Shot Vision-Language Models" (Awadalla et al., 2023)

---

> 📚 **Adaptación al español de la lección [Open Weight VLM Recipes]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).