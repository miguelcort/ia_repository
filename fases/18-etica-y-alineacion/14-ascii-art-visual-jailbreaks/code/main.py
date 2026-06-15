"""
Lección: 14-ascii-art-visual-jailbreaks
Fase: 18
Ética y alineación: 14 Ascii Art Visual Jailbreaks.
"""
from __future__ import annotations
import sys
import numpy as np

def ascii_art_jailbreak(harmful_text, figlet_style="standard"):
    try:
        import pyfiglet
        return pyfiglet.figlet_format(harmful_text, font=figlet_style)
    except ImportError:
        return " ".join(harmful_text)


def visual_prompt_injection(image_with_text, llava_model, target_text):
    prompt = f"Describe text in image. Then: {target_text}"
    return llava_model(image=image_with_text, prompt=prompt)


def ocr_defense(image):
    try:
        import pytesseract
        return pytesseract.image_to_string(image)
    except ImportError:
        return ""



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== 14-ascii-art-visual-jailbreaks ===")
    print(f"Python {sys.version.split()[0]}")
    funcs = ['ascii_art_jailbreak', 'visual_prompt_injection', 'ocr_defense']
    print(f"Funciones disponibles: {len(funcs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
