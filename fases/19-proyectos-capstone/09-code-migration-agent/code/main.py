"""
Lección: 09-code-migration-agent
Fase: 19
Capstone de ingeniería AI: 09 Code Migration Agent.
"""
from __future__ import annotations
import sys

def migrate_code(source_lang, target_lang, source_code, rules, llm):
    """Migrate code: parse AST, translate, test."""
    import tree_sitter
    tree = tree_sitter.Parser(source_lang).parse(source_code)
    translated = translate_ast(tree, target_lang, rules, llm)
    tests = generate_tests(source_code, translated, llm)
    return translated, tests


def translate_ast(tree, target_lang, rules, llm):
    """Translate AST nodes per rules."""
    return llm.translate(tree, target=target_lang, rules=rules)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
