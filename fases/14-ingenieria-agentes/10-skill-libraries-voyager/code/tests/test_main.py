"""Pruebas para 10-skill-libraries-voyager."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestSkill(unittest.TestCase):
    def test_basic(self):
        s = main.Skill("add", "def add(): pass", "Add numbers")
        self.assertEqual(s.name, "add")
        self.assertEqual(s.uses, 0)

    def test_use(self):
        s = main.Skill("a", "code", "desc")
        s.use()
        s.use()
        self.assertEqual(s.uses, 2)

    def test_to_dict(self):
        s = main.Skill("a", "code", "desc", dependencies=["b"])
        d = s.to_dict()
        self.assertEqual(d["name"], "a")
        self.assertEqual(d["dependencies"], ["b"])


class TestSkillLibrary(unittest.TestCase):
    def setUp(self):
        self.lib = main.SkillLibrary(max_size=3)

    def test_add(self):
        self.lib.add_skill(main.Skill("a", "code", "desc"))
        self.assertIn("a", self.lib.skills)

    def test_max_size(self):
        for i in range(5):
            self.lib.add_skill(main.Skill(f"s{i}", "code", "desc"))
        self.assertEqual(len(self.lib.skills), 3)

    def test_get_skill(self):
        s = main.Skill("a", "code", "desc")
        self.lib.add_skill(s)
        self.lib.get_skill("a")
        self.lib.get_skill("a")
        self.assertEqual(s.uses, 2)

    def test_get_missing(self):
        self.assertIsNone(self.lib.get_skill("missing"))

    def test_search(self):
        self.lib.add_skill(main.Skill("add", "code", "Add numbers"))
        self.lib.add_skill(main.Skill("multiply", "code", "Multiply numbers"))
        results = self.lib.search_skills("add")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "add")

    def test_most_used(self):
        for name in ("a", "b", "c"):
            self.lib.add_skill(main.Skill(name, "code", "desc"))
        for _ in range(3):
            self.lib.get_skill("a")
        for _ in range(2):
            self.lib.get_skill("b")
        top = self.lib.most_used(2)
        self.assertEqual(top[0].name, "a")
        self.assertEqual(top[1].name, "b")

    def test_compose(self):
        self.lib.add_skill(main.Skill("a", "def a(): pass", "Func A"))
        self.lib.add_skill(main.Skill("b", "def b(): pass", "Func B"))
        pipeline = self.lib.compose(["a", "b"])
        self.assertIn("def a()", pipeline)
        self.assertIn("def b()", pipeline)


class TestGenerateSkill(unittest.TestCase):
    def test_basic(self):
        s = main.generate_skill_from_code("foo", "code", "desc")
        self.assertEqual(s.name, "foo")


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