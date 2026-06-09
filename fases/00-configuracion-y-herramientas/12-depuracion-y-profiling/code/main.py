"""
Lección: 12-depuracion-y-profiling
Fase: 00
Prerrequisitos: 01-entorno-desarrollo, 10-terminal-y-shell
Fuentes:
- pdb: https://docs.python.org/3/library/pdb.html
- cProfile: https://docs.python.org/3/library/profile.html
- timeit: https://docs.python.org/3/library/timeit.html
- tracemalloc: https://docs.python.org/3/library/tracemalloc.html
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pdb
import pstats
import sys
import timeit
import tracemalloc


def funcion_lenta(n: int) -> int:
    total = 0
    for i in range(n):
        for j in range(n):
            total += i + j
    return total


def version_rapida(n: int) -> int:
    # sum_{i=0}^{n-1} sum_{j=0}^{n-1} (i+j) = n^2 * (n-1)
    return n * n * (n - 1)


def micro_benchmark(n: int, repeticiones: int = 5) -> dict:
    tiempo_lenta = timeit.timeit(lambda: funcion_lenta(n), number=repeticiones)
    tiempo_rapida = timeit.timeit(lambda: version_rapida(n), number=repeticiones)
    return {
        "n": n,
        "repeticiones": repeticiones,
        "lenta_s": tiempo_lenta,
        "rapida_s": tiempo_rapida,
        "speedup": tiempo_lenta / tiempo_rapida if tiempo_rapida > 0 else 0,
    }


def perfilar(n: int, top: int = 5) -> str:
    pr = cProfile.Profile()
    pr.enable()
    funcion_lenta(n)
    pr.disable()
    buf = io.StringIO()
    stats = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    stats.print_stats(top)
    return buf.getvalue()


def memoria(n: int) -> dict:
    tracemalloc.start()
    funcion_lenta(n)
    actual, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "bytes_actual": actual,
        "bytes_pico": pico,
        "kb_pico": pico / 1024,
    }


def depurar(n: int) -> str:
    buf = io.StringIO()
    class StringIOPdb(pdb.Pdb):
        def __init__(self):
            super().__init__(stdout=buf, stdin=io.StringIO("c\n"))
    StringIOPdb().run("funcion_lenta(1)", globals={"funcion_lenta": funcion_lenta})
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description="Demo 12-depuracion-y-profiling")
    parser.add_argument("--n", type=int, default=100, help="Tamano del benchmark")
    parser.add_argument(
        "--modo",
        choices=["benchmark", "profile", "memory", "debug"],
        default="benchmark",
    )
    args = parser.parse_args(argv)
    if args.modo == "benchmark":
        import json
        print(json.dumps(micro_benchmark(args.n), indent=2))
    elif args.modo == "profile":
        print(perfilar(args.n))
    elif args.modo == "memory":
        import json
        print(json.dumps(memoria(args.n), indent=2))
    elif args.modo == "debug":
        print(depurar(args.n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
