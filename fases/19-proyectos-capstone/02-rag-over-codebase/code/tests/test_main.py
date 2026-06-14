"""Pruebas para el capstone 02 — RAG over codebase."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402


class TestChunking(unittest.TestCase):
    def test_chunk_basico(self):
        src = (
            "def foo():\n"
            "    return 1\n\n"
            "def bar():\n"
            "    return 2\n"
        )
        chunks = main.chunk_python_file(src)
        symbols = [c["symbol"] for c in chunks]
        self.assertIn("foo", symbols)
        self.assertIn("bar", symbols)

    def test_chunk_clase(self):
        src = "class Foo:\n    def bar(self):\n        return 1\n"
        chunks = main.chunk_python_file(src)
        symbols = [c["symbol"] for c in chunks]
        self.assertIn("Foo", symbols)
        self.assertIn("bar", symbols)

    def test_chunk_module(self):
        chunks = main.chunk_python_file("x = 1\ny = 2")
        self.assertEqual(chunks[0]["symbol"], "<module>")


class TestTokenize(unittest.TestCase):
    def test_minusculas(self):
        self.assertEqual(main.tokenize("Hola Mundo"), ["hola", "mundo"])

    def test_solo_palabras(self):
        self.assertEqual(
            main.tokenize("foo bar, baz!"), ["foo", "bar", "baz"]
        )


class TestBM25(unittest.TestCase):
    def test_score_positivo_con_overlap(self):
        q = main.tokenize("retry backoff")
        d = main.tokenize("retry with backoff logic")
        df = {"retry": 1, "backoff": 1, "with": 1, "logic": 1}
        s = main.bm25_score(q, d, avg_dl=4.0, N=1, df=df)
        self.assertGreater(s, 0.0)

    def test_score_cero_sin_overlap(self):
        q = main.tokenize("python")
        d = main.tokenize("javascript")
        df = {"python": 1, "javascript": 1}
        s = main.bm25_score(q, d, avg_dl=1.0, N=2, df=df)
        self.assertEqual(s, 0.0)


class TestEmbed(unittest.TestCase):
    def test_norma_uno(self):
        v = main.dense_embed("foo bar foo", {"foo": 0, "bar": 1})
        norm = sum(x * x for x in v) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_vocab_desconocido(self):
        v = main.dense_embed("zzz", {"foo": 0, "bar": 1})
        norm = sum(x * x for x in v) ** 0.5
        self.assertAlmostEqual(norm, 0.0, places=5)


class TestCosine(unittest.TestCase):
    def test_identidad(self):
        v = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(main.cosine(v, v), 1.0, places=5)

    def test_ortogonales(self):
        self.assertAlmostEqual(
            main.cosine([1.0, 0.0], [0.0, 1.0]), 0.0
        )


class TestRerank(unittest.TestCase):
    def test_score_mayor_con_overlap(self):
        s1 = main.rerank("retry", "retry with backoff")
        s2 = main.rerank("retry", "invoice total")
        self.assertGreater(s1, s2)


class TestCodeRAG(unittest.TestCase):
    def test_indexa_archivos(self):
        rag = main.CodeRAG()
        rag.index({"a.py": "def foo():\n    return 1\n"})
        self.assertGreater(len(rag.chunks), 0)
        self.assertGreater(len(rag.embeddings), 0)
        self.assertGreater(len(rag.vocab), 0)

    def test_ask_devuelve_respuesta(self):
        rag = main.CodeRAG()
        rag.index({
            "x.py": "def retry():\n    return 1\n",
            "y.py": "def charge():\n    return 2\n",
        })
        answer = rag.ask("retry", top_k=1)
        self.assertIn("Q: retry", answer)
        self.assertIn("A:", answer)
        self.assertIn("x.py", answer)


class TestMain(unittest.TestCase):
    def test_main_retorna_cero(self):
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        salida = buffer.getvalue()
        self.assertIn("indexados", salida)
        self.assertIn("Q:", salida)


if __name__ == "__main__":
    unittest.main()
