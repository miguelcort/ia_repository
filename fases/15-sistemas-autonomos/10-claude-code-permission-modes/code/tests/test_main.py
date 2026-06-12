"""Pruebas para 10-claude-code-permission-modes."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestPermissionModes(unittest.TestCase):
    def test_list_modes(self):
        modes = main.list_permission_modes()
        self.assertIn("default", modes)
        self.assertIn("acceptEdits", modes)
        self.assertIn("plan", modes)
        self.assertIn("dontAsk", modes)
        self.assertIn("bypassPermissions", modes)

    def test_get_mode(self):
        m = main.get_mode("default")
        self.assertIsNotNone(m)
        self.assertIn("bash", m["tools_require_approval"])

    def test_get_unknown(self):
        self.assertIsNone(main.get_mode("unknown"))

    def test_default_bash_requires(self):
        self.assertTrue(main.requires_approval("default", "bash"))

    def test_default_read_does_not(self):
        self.assertFalse(main.requires_approval("default", "read"))

    def test_acceptEdits_write_auto(self):
        self.assertFalse(main.requires_approval("acceptEdits", "write"))

    def test_acceptEdits_bash_requires(self):
        self.assertTrue(main.requires_approval("acceptEdits", "bash"))

    def test_plan_blocks_all(self):
        self.assertTrue(main.requires_approval("plan", "read"))
        self.assertTrue(main.requires_approval("plan", "bash"))

    def test_dontAsk_denies(self):
        self.assertTrue(main.requires_approval("dontAsk", "read"))
        self.assertTrue(main.requires_approval("dontAsk", "bash"))

    def test_bypassPermissions(self):
        self.assertFalse(main.requires_approval("bypassPermissions", "bash"))
        self.assertFalse(main.requires_approval("bypassPermissions", "rm"))

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            main.requires_approval("unknown", "bash")


class TestPlanStep(unittest.TestCase):
    def test_plan_step(self):
        steps = ["step1", "step2", "step3"]
        self.assertEqual(main.plan_step(steps, 0), "step1")
        self.assertEqual(main.plan_step(steps, 1), "step2")
        self.assertIsNone(main.plan_step(steps, 3))


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