"""
Lección: 20-salidas-estructuradas-y-decoding-constrenido
Fase: 05
Prerrequisitos: 19-tokenizacion-de-subpalabras
"""
from __future__ import annotations
import sys
import re
import numpy as np


def forzar_json(texto, schema_claves):
    """Fuerza que la salida sea JSON con las claves requeridas.
    Mock: extrae pares clave:valor del texto."""
    resultado = {}
    for clave in schema_claves:
        patron = rf"{clave}\s*[:=]\s*([\w\d.\-]+)"
        match = re.search(patron, texto, re.IGNORECASE)
        if match:
            valor = match.group(1)
            # Intentar convertir a numero
            try:
                resultado[clave] = float(valor)
            except ValueError:
                resultado[clave] = valor
        else:
            resultado[clave] = None
    return resultado


def validar_esquema(objeto, schema):
    """Valida que objeto tiene las claves y tipos del schema."""
    if not isinstance(objeto, dict):
        return False
    for clave, tipo_esperado in schema.items():
        if clave not in objeto:
            return False
        if not isinstance(objeto[clave], tipo_esperado):
            return False
    return True


def grammar_constrained_decoding(tokens_validos, logits):
    """Mock: forzar al modelo a solo generar tokens validos.
    Devuelve logits con -inf en tokens invalidos."""
    mask = np.full_like(logits, -1e9)
    for t in tokens_validos:
        if 0 <= t < len(mask):
            mask[t] = logits[t]
    return mask


def main() -> int:
    schema = {"nombre": str, "edad": float, "ciudad": str}
    texto = "nombre: Maria, edad: 30, ciudad: Madrid"
    obj = forzar_json(texto, schema.keys())
    print(f"Parseado: {obj}")
    print(f"Valido: {validar_esquema(obj, schema)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())