# 14 — ASCII art y visual jailbreaks

> ASCII art jailbreaks (Glm 2024): un modelo puede "ver" texto en arte ASCII. Prompt injection visual: código en imágenes que evade text-only safety filters. Multimodal models son vulnerables.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 18/12, 18/13
**Tiempo estimado:** ~25 minutos

## Objetivos

- Generar ASCII art de harmful text.
- Evaluar visual jailbreak success.
- Implementar OCR defense.
- Diagnosticar multimodal attack surface.

## Constrúyelo

```python
import numpy as np


def ascii_art_jailbreak(harmful_text, figlet_style="standard"):
    """Genera ASCII art del harmful text."""
    try:
        import pyfiglet
        return pyfiglet.figlet_format(harmful_text, font=figlet_style)
    except ImportError:
        return " ".join(harmful_text)


def visual_prompt_injection(image_with_text, llava_model,
                          target_text):
    """Prompt injection visual: texto en imagen evade
    text safety filter."""
    prompt = f"Describe text in image. Then: {target_text}"
    return llava_model(image=image_with_text, prompt=prompt)


def ocr_defense(image):
    """OCR defense: extrae texto, aplica text safety filter."""
    import pytesseract
    text = pytesseract.image_to_string(image)
    return text
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-visual-jailbreak
fase: 18
leccion: 14
---

1. Generar ASCII art de harmful text.
2. Evaluar text-only safety filter.
3. Implementar OCR defense.
4. Multimodal attack surface.
```

## Ejercicios

1. **ASCII art**: probar figlet fonts en
   frontier model.
2. **Visual injection**: texto oculto en
   imagen, evaluar LLaVA.
3. **Desafío**: diseñar multimodal defense.

## Lecturas recomendadas

- "ArtPrompt: ASCII Art-based Jailbreak" (Glm 2024)
- "Visual Prompt Injection" (Microsoft 2024)

---

> 📚 **Adaptación al español** de la lección
> "[14-ascii-art-visual-jailbreaks]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
