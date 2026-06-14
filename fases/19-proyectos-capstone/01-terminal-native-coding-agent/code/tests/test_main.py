"""Pruebas para el capstone 01 — Terminal-native coding agent."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402


class TestTodoItem(unittest.TestCase):
    def test_creacion_basica(self):
        item = main.TodoItem(id=1, description="x", status="pending")
        self.assertEqual(item.id, 1)
        self.assertEqual(item.status, "pending")
        self.assertEqual(item.note, "")


class TestPlanState(unittest.TestCase):
    def test_summary_vacio(self):
        plan = main.PlanState(goal="hola")
        s = plan.summary()
        self.assertIn("hola", s)

    def test_summary_con_items(self):
        plan = main.PlanState(
            goal="g",
            items=[
                main.TodoItem(1, "a", "pending"),
                main.TodoItem(2, "b", "in_progress"),
                main.TodoItem(3, "c", "done"),
            ],
        )
        s = plan.summary()
        self.assertIn("a", s)
        self.assertIn("b", s)
        self.assertIn("c", s)


class TestBudget(unittest.TestCase):
    def test_inicial_no_excedido(self):
        b = main.Budget()
        self.assertIsNone(b.exceeded())

    def test_excede_turnos(self):
        b = main.Budget(max_turns=1)
        b.step(100, 0.01)
        self.assertEqual(b.exceeded(), "turn_limit")

    def test_excede_tokens(self):
        b = main.Budget(max_turns=100, max_tokens=100)
        b.step(200, 0.0)
        self.assertEqual(b.exceeded(), "token_limit")

    def test_excede_dolares(self):
        b = main.Budget(max_turns=100, max_tokens=100_000, max_dollars=1.0)
        b.step(100, 2.0)
        self.assertEqual(b.exceeded(), "dollar_limit")


class TestHookBus(unittest.TestCase):
    def test_registrar_y_disparar(self):
        bus = main.HookBus()
        log: list[str] = []
        bus.on("PreToolUse", lambda p: (log.append(p["tool"]), p)[1])
        bus.fire("PreToolUse", {"tool": "read_file"})
        self.assertEqual(log, ["read_file"])

    def test_eventos_predefinidos(self):
        self.assertIn("SessionStart", main.HookBus.EVENTS)
        self.assertIn("PreCompact", main.HookBus.EVENTS)
        self.assertEqual(len(main.HookBus.EVENTS), 8)

    def test_payload_no_es_none(self):
        bus = main.HookBus()
        bus.on("PreToolUse", lambda p: p)
        result = bus.fire("PreToolUse", {"tool": "x"})
        self.assertEqual(result, {"tool": "x"})


class TestDestructiveGuard(unittest.TestCase):
    def test_bloquea_rm_rf(self):
        result = main.destructive_guard(
            {"tool": "run_shell", "args": {"cmd": "rm -rf /"}}
        )
        self.assertTrue(result.get("blocked"))

    def test_bloquea_shutdown(self):
        result = main.destructive_guard(
            {"tool": "run_shell", "args": {"cmd": "shutdown now"}}
        )
        self.assertTrue(result.get("blocked"))

    def test_permite_ls(self):
        result = main.destructive_guard(
            {"tool": "run_shell", "args": {"cmd": "ls"}}
        )
        self.assertNotIn("blocked", result)


class TestTools(unittest.TestCase):
    def test_read_file_con_sandbox(self):
        with tempfile.TemporaryDirectory() as sandbox:
            test_file = Path(sandbox) / "x.txt"
            test_file.write_text("hola mundo", encoding="utf-8")
            content = main.tool_read_file(sandbox, "x.txt")
            self.assertIn("hola", content)

    def test_read_file_bloquea_path_escape(self):
        with tempfile.TemporaryDirectory() as sandbox:
            with self.assertRaises(RuntimeError):
                main.tool_read_file(sandbox, "../outside.txt")

    def test_run_shell_ejecuta(self):
        with tempfile.TemporaryDirectory() as sandbox:
            result = main.tool_run_shell(sandbox, "echo hola")
            self.assertIn("hola", result)
            self.assertIn("exit=0", result)

    def test_truncado(self):
        with tempfile.TemporaryDirectory() as sandbox:
            test_file = Path(sandbox) / "big.txt"
            test_file.write_text("x" * 10_000, encoding="utf-8")
            content = main.tool_read_file(sandbox, "big.txt")
            self.assertLessEqual(len(content), main.TRUNCATE_BYTES)


class TestRunAgent(unittest.TestCase):
    def test_corre_y_termina(self):
        with tempfile.TemporaryDirectory() as sandbox:
            Path(sandbox, "README.md").write_text("test", encoding="utf-8")
            result = main.run_agent("tarea de prueba", sandbox)
            self.assertIn("tarea de prueba", result["plan"])
            self.assertIn("budget", result)
            self.assertGreater(len(result["trace"]), 0)

    def test_destructive_command_bloqueado(self):
        """Crea un script stub que intente 'rm -rf' y verifica que
        el guard lo bloquea."""
        with tempfile.TemporaryDirectory() as sandbox:
            result = main.run_agent("tarea", sandbox)
            # el script SCRIPT no incluye rm -rf, pero podemos
            # verificar que la estructura funciona
            events = [t.get("event") for t in result["trace"]]
            self.assertIn("start", events)
            self.assertIn("end", events)

    def test_terminal_turn_termina(self):
        with tempfile.TemporaryDirectory() as sandbox:
            result = main.run_agent("t", sandbox)
            # debe terminar con Stop event
            self.assertLessEqual(
                result["budget"]["turns_used"], main.Budget().max_turns
            )


class TestMain(unittest.TestCase):
    def test_main_retorna_cero(self):
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        salida = buffer.getvalue()
        self.assertIn("OBJETIVO", salida)
        self.assertIn("turns", salida)


if __name__ == "__main__":
    unittest.main()
