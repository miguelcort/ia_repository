"""Pruebas para 06-hierarchical-architecture."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestHierarchicalNode(unittest.TestCase):
    def test_create(self):
        n = main.HierarchicalNode(role="dev")
        self.assertEqual(n.role, "dev")
        self.assertEqual(n.level, 0)
        self.assertTrue(n.is_leaf())

    def test_add_child(self):
        parent = main.HierarchicalNode()
        child = parent.add_child(main.HierarchicalNode(role="dev"))
        self.assertEqual(child.parent, parent)
        self.assertEqual(child.level, 1)
        self.assertFalse(parent.is_leaf())

    def test_delegate_leaf(self):
        n = main.HierarchicalNode(role="dev")
        r = n.delegate({"action": "code"})
        self.assertIn("leaf:dev:code", r)

    def test_delegate_with_children(self):
        parent = main.HierarchicalNode(role="lead")
        c1 = parent.add_child(main.HierarchicalNode(role="dev1"))
        c2 = parent.add_child(main.HierarchicalNode(role="dev2"))
        r = parent.delegate({"action": "ship"})
        self.assertIn("agg:lead", r)
        self.assertIn("leaf:dev1", r)
        self.assertIn("leaf:dev2", r)

    def test_tree(self):
        parent = main.HierarchicalNode(role="lead")
        parent.add_child(main.HierarchicalNode(role="dev"))
        t = parent.tree()
        self.assertIn("[0] lead", t)
        self.assertIn("[1] dev", t)


class TestBuildTree(unittest.TestCase):
    def test_build(self):
        root = main.build_tree()
        self.assertEqual(root.role, "manager")
        self.assertEqual(len(root.children), 2)
        self.assertEqual(root.children[0].role, "tech_lead")

    def test_delegate_tree(self):
        root = main.build_tree()
        r = root.delegate({"action": "ship"})
        self.assertIn("manager", r)
        self.assertIn("developer", r)
        self.assertIn("tester", r)


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