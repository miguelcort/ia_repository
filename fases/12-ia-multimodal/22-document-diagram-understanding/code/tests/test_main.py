"""Pruebas para 22-document-diagram-understanding."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestOCRExtract(unittest.TestCase):
    def test_shape(self):
        img = np.random.default_rng(0).standard_normal((512, 512, 3))
        lines = main.ocr_extract_lines(img, n_lines=10)
        self.assertEqual(len(lines), 10)
        for line in lines:
            self.assertIn("text", line)
            self.assertIn("bbox", line)
            x0, y0, x1, y1 = line["bbox"]
            self.assertGreater(x1, x0)
            self.assertGreater(y1, y0)


class TestLayout(unittest.TestCase):
    def test_header(self):
        img = np.random.default_rng(0).standard_normal((512, 512, 3))
        lines = main.ocr_extract_lines(img, n_lines=5)
        # override first line to be in header zone
        lines[0]["bbox"] = (0, 0, 100, 20)
        regions = main.layout_analysis(lines, img.shape)
        # first region should be header
        self.assertEqual(regions[0]["type"], "header")


class TestTableExtract(unittest.TestCase):
    def test_table(self):
        img = np.random.default_rng(0).standard_normal((512, 512, 3))
        tables = main.table_extract([], img.shape)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["type"], "table")


class TestChart(unittest.TestCase):
    def test_chart(self):
        img = np.random.default_rng(0).standard_normal((512, 512, 3))
        charts = main.chart_extract(img)
        self.assertEqual(len(charts), 1)
        self.assertEqual(charts[0]["type"], "chart")


class TestDocVQA(unittest.TestCase):
    def test_answer(self):
        img = np.random.default_rng(0).standard_normal((512, 512, 3))
        lines = main.ocr_extract_lines(img, n_lines=3)
        ans = main.document_vqa(img, "What is the title?", lines)
        self.assertIn("Question", ans)
        self.assertIn("OCR", ans)


class TestLayoutLMv3(unittest.TestCase):
    def test_shape(self):
        img = np.random.default_rng(0).standard_normal((512, 512, 3))
        lines = main.ocr_extract_lines(img, n_lines=5)
        feats = main.layoutlmv3_encode(img, lines, embed_dim=768)
        # 5 lines + 1 CLS = 6
        self.assertEqual(feats.shape, (6, 768))


class TestNougat(unittest.TestCase):
    def test_paper(self):
        img = np.random.default_rng(0).standard_normal((512, 512, 3))
        md = main.nougat_scientific_paper(img)
        self.assertIn("#", md)
        self.assertIn("Abstract", md)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()