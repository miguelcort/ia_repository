"""Pruebas para la lección. Mínimo 5."""
import unittest


class TestLeccion(unittest.TestCase):
    def test_placeholder(self):
        """Reemplazar con la primera prueba real."""
        self.assertTrue(True)

    def test_segunda(self):
        self.assertEqual(1 + 1, 2)

    def test_tercera(self):
        self.assertIn("a", "abc")

    def test_cuarta(self):
        items = [1, 2, 3]
        self.assertEqual(len(items), 3)

    def test_quinta(self):
        d = {"a": 1}
        self.assertEqual(d["a"], 1)


if __name__ == "__main__":
    unittest.main()
