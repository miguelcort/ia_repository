"""Pruebas para 08-role-specialization."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestRoleTemplates(unittest.TestCase):
    def test_list(self):
        roles = main.list_roles()
        self.assertIn("developer", roles)
        self.assertIn("tester", roles)
        self.assertIn("reviewer", roles)
        self.assertIn("researcher", roles)
        self.assertIn("product_manager", roles)

    def test_get(self):
        r = main.get_role("developer")
        self.assertIn("system_prompt", r)
        self.assertIn("tools", r)
        self.assertIn("temperature", r)

    def test_get_unknown(self):
        self.assertIsNone(main.get_role("unknown"))


class TestSpecialistAgent(unittest.TestCase):
    def test_create(self):
        a = main.SpecialistAgent("developer")
        self.assertEqual(a.role_name, "developer")
        self.assertEqual(a.name, "Developer")
        self.assertIsNotNone(a.agent_id)

    def test_create_unknown_raises(self):
        with self.assertRaises(ValueError):
            main.SpecialistAgent("unknown")

    def test_custom_prompt(self):
        a = main.SpecialistAgent("developer", custom_prompt="Custom")
        self.assertEqual(a.system_prompt, "Custom")

    def test_can_use(self):
        a = main.SpecialistAgent("developer")
        self.assertTrue(a.can_use("write"))
        self.assertTrue(a.can_use("bash"))
        self.assertFalse(a.can_use("deploy"))

    def test_describe(self):
        a = main.SpecialistAgent("tester")
        d = a.describe()
        self.assertEqual(d["role"], "tester")
        self.assertIn("bash", d["tools"])
        self.assertEqual(d["temperature"], 0.1)


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