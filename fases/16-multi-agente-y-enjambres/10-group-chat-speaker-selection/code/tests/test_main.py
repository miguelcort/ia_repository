"""Pruebas para 10-group-chat-speaker-selection."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestRoundRobin(unittest.TestCase):
    def test_basic(self):
        idx, speaker = main.round_robin(["a", "b", "c"], 0, 1)
        self.assertEqual(speaker, "b")
        self.assertEqual(idx, 1)

    def test_wrap(self):
        idx, speaker = main.round_robin(["a", "b", "c"], 2, 1)
        self.assertEqual(speaker, "a")
        self.assertEqual(idx, 0)

    def test_empty(self):
        idx, speaker = main.round_robin([], 0, 1)
        self.assertIsNone(speaker)


class TestRandom(unittest.TestCase):
    def test_basic(self):
        import random
        rng = random.Random(42)
        speaker = main.random_speaker(["a", "b", "c"], rng=rng)
        self.assertIn(speaker, ["a", "b", "c"])

    def test_empty(self):
        self.assertIsNone(main.random_speaker([]))


class TestModerator(unittest.TestCase):
    def test_moderator_speaks(self):
        m = main.moderator_speaker(["a1", "a2", "a3"], moderator="mod", last_speaker="a1")
        self.assertEqual(m, "mod")

    def test_advance(self):
        m = main.moderator_speaker(["a1", "a2", "mod"], moderator="mod", last_speaker="mod")
        self.assertEqual(m, "a1")


class TestWeighted(unittest.TestCase):
    def test_basic(self):
        import random
        rng = random.Random(42)
        speaker = main.weighted_speaker(["a", "b", "c"], [1, 0, 0], rng=rng)
        self.assertEqual(speaker, "a")

    def test_empty(self):
        self.assertIsNone(main.weighted_speaker([], []))


class TestTopic(unittest.TestCase):
    def test_match(self):
        expertise = {"a1": ["code"], "a2": ["test"], "a3": ["design"]}
        speaker = main.topic_speaker(["a1", "a2", "a3"], {"topic": "code", **expertise})
        self.assertEqual(speaker, "a1")

    def test_no_topic(self):
        speaker = main.topic_speaker(["a1", "a2"], {"a1": [], "a2": []})
        self.assertIn(speaker, ["a1", "a2"])


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