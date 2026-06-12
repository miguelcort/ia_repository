"""Pruebas para 14-consensus-and-bft."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestMajority(unittest.TestCase):
    def test_clear_majority(self):
        d, c = main.majority_vote(["yes", "yes", "no", "yes"])
        self.assertEqual(d, "yes")
        self.assertEqual(c, 3)

    def test_tie(self):
        d, c = main.majority_vote(["a", "b"])
        self.assertIsNone(d)
        self.assertEqual(c, 1)

    def test_empty(self):
        d, c = main.majority_vote([])
        self.assertIsNone(d)
        self.assertEqual(c, 0)


class TestQuorum(unittest.TestCase):
    def test_quorum_reached(self):
        d, c = main.quorum_decide(["a", "a", "b", "c"], quorum_size=3)
        self.assertEqual(d, "a")

    def test_quorum_not_reached(self):
        d, c = main.quorum_decide(["a", "b"], quorum_size=3)
        self.assertIsNone(d)

    def test_empty(self):
        d, _ = main.quorum_decide([], quorum_size=2)
        self.assertIsNone(d)


class TestPBFT(unittest.TestCase):
    def test_4_nodes_1_byzantine(self):
        d, err = main.pbft_consensus(["x", "x", "x", "x"], byzantine_tolerance=1)
        self.assertEqual(d, "x")

    def test_insufficient_nodes(self):
        d, err = main.pbft_consensus(["x", "x", "x"], byzantine_tolerance=1)
        self.assertIsNone(d)
        self.assertEqual(err, "insufficient_nodes")

    def test_7_nodes_2_byzantine(self):
        d, _ = main.pbft_consensus(["x"] * 5 + ["y", "z"], byzantine_tolerance=2)
        self.assertEqual(d, "x")


class TestRaft(unittest.TestCase):
    def test_leader_accepted(self):
        d, c, lid = main.raft_consensus("x", ["x", "x"], leader_id="L1")
        self.assertEqual(d, "x")
        self.assertEqual(lid, "L1")

    def test_leader_rejected(self):
        d, _, _ = main.raft_consensus("x", ["y", "y", "y"], leader_id="L1")
        self.assertIsNone(d)


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