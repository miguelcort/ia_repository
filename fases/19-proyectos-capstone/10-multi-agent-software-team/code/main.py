"""
Lección: 10-multi-agent-software-team
Fase: 19
Capstone de ingeniería AI: 10 Multi Agent Software Team.
"""
from __future__ import annotations
import sys

def build_software_team():
    """Build crew: PM, architect, dev, QA."""
    from crewai import Agent, Crew, Task
    pm = Agent(role="ProductManager", goal="Spec the app")
    arch = Agent(role="Architect", goal="Design system")
    dev = Agent(role="Developer", goal="Write code")
    qa = Agent(role="QA", goal="Test thoroughly")
    return Crew(agents=[pm, arch, dev, qa],
               tasks=[Task("Write spec"), Task("Design"),
                     Task("Implement"), Task("Test")])



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
