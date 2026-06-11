"""
Lección: 19-ocr-y-comprension-de-documentos
Fase: 04
Prerrequisitos: 18-clip-vocabulario-abierto
"""
from __future__ import annotations
import sys
import numpy as np


def preprocesar_ocr(img, binario=True, umbral=128):
    """Binariza imagen para OCR."""
    if img.ndim == 3:
        gris = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.uint8)
    else:
        gris = img
    if binario:
        return (gris > umbral).astype(np.uint8) * 255
    return gris


def proyectar_horizontal(img_bin):
    """Suma de pixeles 'activos' (no blancos) por fila -> picos son lineas."""
    return ((255 - img_bin).astype(np.int32)).sum(axis=1)


def proyectar_vertical(img_bin):
    """Suma de pixeles 'activos' por columna -> picos son letras."""
    return ((255 - img_bin).astype(np.int32)).sum(axis=0)


def segmentos_por_huecos(proyeccion, min_gap=2):
    """Encuentra segmentos (lineas) en la proyeccion separados por huecos."""
    segmentos = []
    in_segmento = False
    start = 0
    gap_count = 0
    for i, val in enumerate(proyeccion):
        if val > 0:
            if not in_segmento:
                start = i
                in_segmento = True
            gap_count = 0
        else:
            if in_segmento:
                gap_count += 1
                if gap_count >= min_gap:
                    segmentos.append((start, i - gap_count + 1))
                    in_segmento = False
                    gap_count = 0
    if in_segmento:
        segmentos.append((start, len(proyeccion)))
    return segmentos


def simular_ocr_patch(patch):
    """Mock OCR: devuelve un texto deterministico segun el tamano del patch."""
    if patch.size == 0 or patch.sum() == 0:
        return ""
    h = patch.shape[0]
    if h < 8:
        return ""
    # Texto simulado
    return "Lorem ipsum " + str(h)


def main() -> int:
    # Imagen con 'texto' simulado (rectangulos)
    img = np.ones((50, 200), dtype=np.uint8) * 255
    # Linea 1: pixeles negros
    img[10:20, 20:60] = 0
    img[10:20, 70:110] = 0
    img[10:20, 120:180] = 0
    # Linea 2
    img[30:40, 30:50] = 0
    img[30:40, 60:100] = 0
    img[30:40, 110:160] = 0
    binaria = preprocesar_ocr(img)
    proj_h = proyectar_horizontal(binaria)
    lineas = segmentos_por_huecos(proj_h, min_gap=2)
    print(f"Lineas detectadas: {lineas}")
    for li, lf in lineas:
        texto = simular_ocr_patch(img[li:lf])
        print(f"  Linea {li}-{lf}: '{texto}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())