"""
Lección: 11-multi-region-kv-locality
Fase: 17
Multi-region KV cache locality: route
requests to nearest replica, replicate
KV cache, sticky routing, regional
compliance.
"""
from __future__ import annotations


class Region:
    def __init__(self, name, location, replicas=1):
        self.name = name
        self.location = location
        self.replicas = replicas
        self.kv_cache = {}


class KVLocalityRouter:
    def __init__(self):
        self.regions = {}
        self.user_region = {}

    def add_region(self, region):
        self.regions[region.name] = region

    def assign_user(self, user_id, region_name):
        if region_name in self.regions:
            self.user_region[user_id] = region_name

    def route(self, user_id):
        """Return the region for a user, fallback to first available."""
        region_name = self.user_region.get(user_id)
        if region_name and region_name in self.regions:
            return self.regions[region_name]
        if self.regions:
            return next(iter(self.regions.values()))
        return None

    def get_cache(self, user_id, key):
        region = self.route(user_id)
        if region is None:
            return None
        return region.kv_cache.get(key)

    def set_cache(self, user_id, key, value):
        region = self.route(user_id)
        if region is None:
            return False
        region.kv_cache[key] = value
        return True

    def replicate_to(self, from_user, to_region, key, value):
        if to_region in self.regions:
            self.regions[to_region].kv_cache[key] = value


def nearest_region(user_location, regions):
    """Pick the region with matching location, else first."""
    for r in regions.values():
        if r.location == user_location:
            return r
    if regions:
        return next(iter(regions.values()))
    return None


def main() -> int:
    router = KVLocalityRouter()
    router.add_region(Region("us-east", "US"))
    router.add_region(Region("eu-west", "EU"))
    router.assign_user("user1", "us-east")
    print(router.route("user1").name)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())