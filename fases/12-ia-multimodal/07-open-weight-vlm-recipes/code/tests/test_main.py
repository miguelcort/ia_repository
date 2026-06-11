"""Pruebas para 07-open-weight-vlm-recipes."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestListRecipes(unittest.TestCase):
    def test_minimum_recipes(self):
        recipes = main.list_recipes()
        self.assertGreaterEqual(len(recipes), 5)

    def test_known_recipes(self):
        recipes = main.list_recipes()
        for name in ("llava-1.5", "llava-next", "idefics2", "open-flamingo", "molmo"):
            self.assertIn(name, recipes)


class TestGetRecipe(unittest.TestCase):
    def test_llava_1_5(self):
        r = main.get_recipe("llava-1.5")
        self.assertEqual(r["name"], "LLaVA-1.5")
        self.assertIn("Vicuna", r["llm"])

    def test_unknown(self):
        r = main.get_recipe("unknown-vlm")
        self.assertIsNone(r)


class TestRecipesByParams(unittest.TestCase):
    def test_7b_to_15b(self):
        recipes = main.recipes_by_param_count(min_b=7, max_b=15)
        # Idefics2 8B, OpenFlamingo 9B, Molmo 7B
        names = [r["name"] for r in recipes]
        self.assertIn("Idefics2", names)
        self.assertIn("Molmo", names)

    def test_30b_plus(self):
        recipes = main.recipes_by_param_count(min_b=30, max_b=70)
        # LLaVA-Next 34B
        self.assertGreaterEqual(len(recipes), 1)


class TestTrainingStage(unittest.TestCase):
    def test_pretrain(self):
        d = main.training_stage_to_data("pretrain_projector")
        self.assertGreater(len(d), 0)


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