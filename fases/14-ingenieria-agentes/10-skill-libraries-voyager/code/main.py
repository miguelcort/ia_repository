"""
Lección: 10-skill-libraries-voyager
Fase: 14
Voyager (Wang 2023): skill library para LLM agents.
Skills = reusable code (functions, classes) learned via
curiosity-driven exploration. GPT-4 generates skills.
"""
from __future__ import annotations
import time


class Skill:
    """A reusable skill (code + description)."""
    def __init__(self, name, code, description, dependencies=None):
        self.name = name
        self.code = code
        self.description = description
        self.dependencies = dependencies or []
        self.uses = 0
        self.created_at = time.time()

    def use(self):
        """Record a use."""
        self.uses += 1

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "code": self.code,
            "dependencies": self.dependencies,
            "uses": self.uses,
        }


class SkillLibrary:
    """Mock Voyager skill library."""
    def __init__(self, max_size=100):
        self.skills = {}
        self.max_size = max_size

    def add_skill(self, skill):
        """Add skill. Evict least used if full."""
        if len(self.skills) >= self.max_size:
            # evict least used
            lru = min(self.skills.values(), key=lambda s: (s.uses, -s.created_at))
            del self.skills[lru.name]
        self.skills[skill.name] = skill

    def get_skill(self, name):
        s = self.skills.get(name)
        if s:
            s.use()
        return s

    def search_skills(self, query):
        """Search skills by query in name or description."""
        results = []
        for skill in self.skills.values():
            if query.lower() in skill.name.lower() or query.lower() in skill.description.lower():
                results.append(skill)
        return results

    def list_skills(self):
        return list(self.skills.values())

    def most_used(self, n=3):
        return sorted(self.skills.values(), key=lambda s: s.uses, reverse=True)[:n]

    def compose(self, skill_names):
        """Compose skills into a pipeline."""
        composed_code = "\n\n".join(
            f"# {s.description}\n{s.code}" for s in (self.get_skill(n) for n in skill_names) if s
        )
        return composed_code


def generate_skill_from_code(name, code, description):
    """Create a skill from code (mock LLM generation)."""
    return Skill(name=name, code=code, description=description)


def main() -> int:
    lib = SkillLibrary()
    lib.add_skill(Skill("add", "def add(a,b): return a+b", "Add two numbers"))
    lib.add_skill(Skill("multiply", "def multiply(a,b): return a*b", "Multiply two numbers"))
    lib.get_skill("add")
    lib.get_skill("add")
    lib.get_skill("multiply")
    top = lib.most_used(2)
    print(f"Most used: {[(s.name, s.uses) for s in top]}")
    pipeline = lib.compose(["add", "multiply"])
    print(f"Pipeline: {len(pipeline)} chars")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())