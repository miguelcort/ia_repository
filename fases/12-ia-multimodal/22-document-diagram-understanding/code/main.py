"""
Lección: 22-document-diagram-understanding
Fase: 12
Document + diagram understanding: OCR + layout analysis + VQA.
DocVLM, LayoutLMv3, Donut, Nougat, GOT, SmolDocling, DocOwl2.
PDFs, forms, tables, charts, scientific papers.
"""
from __future__ import annotations
import numpy as np


def ocr_extract_lines(image, n_lines=10):
    """Mock OCR: image -> (n_lines,) text + bboxes."""
    rng = np.random.default_rng(hash(image.tobytes()[:64]) & 0xFFFFFFFF)
    H, W, _ = image.shape
    lines = []
    for i in range(n_lines):
        x0 = rng.integers(0, W - 100)
        y0 = int(i * H / n_lines)
        text = f"line {i}"
        lines.append({"text": text, "bbox": (x0, y0, x0 + 100, y0 + 20)})
    return lines


def layout_analysis(ocr_lines, image_shape):
    """Layout: detect regions (text, table, figure, header)."""
    H, W = image_shape[:2]
    regions = []
    for i, line in enumerate(ocr_lines):
        x0, y0, x1, y1 = line["bbox"]
        # clasificar: top -> header, middle -> text/table/figure
        if y0 < H * 0.1:
            rtype = "header"
        elif y0 < H * 0.5:
            rtype = "text"
        else:
            rtype = "text"
        regions.append({"type": rtype, "bbox": (x0, y0, x1, y1), "text": line["text"]})
    return regions


def table_extract(ocr_lines, image_shape):
    """Detect tables via grid structure (mock)."""
    H, W = image_shape[:2]
    # mock: detect 1 table at center
    return [{"type": "table", "bbox": (W // 4, H // 3, 3 * W // 4, 2 * H // 3), "rows": []}]


def chart_extract(image):
    """Detect chart (mock)."""
    H, W, _ = image.shape
    return [{"type": "chart", "bbox": (W // 4, H // 4, 3 * W // 4, 3 * H // 4)}]


def document_vqa(image, question, ocr_lines):
    """Document VQA: combine OCR + layout + question."""
    context = " ".join([l["text"] for l in ocr_lines])
    return f"Based on OCR: {context}. Question: {question}"


def layoutlmv3_encode(image, ocr_lines, embed_dim=768):
    """LayoutLMv3: image + text + layout (x, y, w, h) -> (n, embed_dim)."""
    rng = np.random.default_rng(0)
    n = len(ocr_lines) + 1  # +1 for image CLS
    return rng.standard_normal((n, embed_dim)) * 0.1


def nougat_scientific_paper(image):
    """Nougat: paper -> markdown (mock)."""
    return "# Paper\n\nAbstract: ...\n\n## Methods\n..."

def main() -> int:
    rng = np.random.default_rng(0)
    img = rng.standard_normal((1024, 768, 3))
    lines = ocr_extract_lines(img, n_lines=8)
    print(f"OCR lines: {len(lines)}")
    regions = layout_analysis(lines, img.shape)
    print(f"Layout regions: {len(regions)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())