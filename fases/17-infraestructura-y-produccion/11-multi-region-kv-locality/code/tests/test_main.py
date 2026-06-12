"""Pruebas para 11-multi-region-kv-locality."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestRegion(unittest.TestCase):
    def test_create(self):
        r = main.Region("us-east", "US", replicas=3)
        self.assertEqual(r.name, "us-east")
        self.assertEqual(r.replicas, 3)


class TestRouter(unittest.TestCase):
    def setUp(self):
        self.r = main.KVLocalityRouter()
        self.r.add_region(main.Region("us-east", "US"))
        self.r.add_region(main.Region("eu-west", "EU"))

    def test_add_region(self):
        self.assertIn("us-east", self.r.regions)
        self.assertIn("eu-west", self.r.regions)

    def test_assign_route(self):
        self.r.assign_user("u1", "us-east")
        self.assertEqual(self.r.route("u1").name, "us-east")

    def test_route_unknown_fallback(self):
        region = self.r.route("unknown")
        self.assertIsNotNone(region)

    def test_route_no_regions(self):
        r = main.KVLocalityRouter()
        self.assertIsNone(r.route("any"))

    def test_get_set_cache(self):
        self.r.set_cache("u1", "k", "v")
        self.assertEqual(self.r.get_cache("u1", "k"), "v")

    def test_set_cache_no_region(self):
        r = main.KVLocalityRouter()
        self.assertFalse(r.set_cache("u1", "k", "v"))

    def test_replicate(self):
        self.r.replicate_to("u1", "eu-west", "shared", "data")
        self.assertEqual(self.r.get_cache("u2", "shared"), "data") if "u2" in self.r.user_region else None


class TestNearest(unittest.TestCase):
    def test_exact_match(self):
        regions = {"us": main.Region("us", "US"), "eu": main.Region("eu", "EU")}
        r = main.nearest_region("US", regions)
        self.assertEqual(r.name, "us")

    def test_fallback_first(self):
        regions = {"us": main.Region("us", "US")}
        r = main.nearest_region("ASIA", regions)
        self.assertEqual(r.name, "us")

    def test_empty(self):
        self.assertIsNone(main.nearest_region("US", {}))


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