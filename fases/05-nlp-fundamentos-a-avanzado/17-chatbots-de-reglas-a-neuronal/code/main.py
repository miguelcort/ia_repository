"""
Lección: 17-chatbots-de-reglas-a-neuronal
Fase: 05
Prerrequisitos: 16-generacion-de-texto-pre-transformer
"""
from __future__ import annotations
import sys
import re
import numpy as np


def chatbot_reglas(intencion, texto):
    """Chatbot basado en reglas: matching de patrones."""
    reglas = {
        "saludo": re.compile(r"\b(hola|hi|buenos dias|buenas)\b", re.IGNORECASE),
        "despedida": re.compile(r"\b(chao|adios|hasta luego|bye)\b", re.IGNORECASE),
        "nombre": re.compile(r"\b(me llamo|soy|mi nombre)\s+(\w+)", re.IGNORECASE),
    }
    for intencion_regla, patron in reglas.items():
        if patron.search(texto):
            if intencion_regla == "nombre":
                match = patron.search(texto)
                return f"Encantado, {match.group(2)}"
            elif intencion_regla == "saludo":
                return "Hola! En que puedo ayudarte?"
            elif intencion_regla == "despedida":
                return "Hasta luego!"
    return None


def chatbot_retrieval(pregunta, faq):
    """Chatbot retrieval: encuentra la pregunta FAQ mas similar."""
    q_tokens = set(re.findall(r"\b\w+\b", pregunta.lower()))
    mejor = None
    mejor_score = 0
    for q_faq, respuesta in faq.items():
        f_tokens = set(re.findall(r"\b\w+\b", q_faq.lower()))
        if not q_tokens or not f_tokens:
            continue
        overlap = len(q_tokens & f_tokens) / len(q_tokens | f_tokens)
        if overlap > mejor_score:
            mejor_score = overlap
            mejor = respuesta
    return mejor, mejor_score


def chatbot_generativo_mock(prompt):
    """Mock: chatbot neuronal. Devuelve respuesta 'creativa' deterministica."""
    rng = np.random.default_rng(hash(prompt) % 2**32)
    respuestas = [
        "Interesante pregunta. En mi opinion, ",
        "Eso depende del contexto. Generalmente, ",
        "Te cuento: ",
    ]
    continuacion = ["si, exactamente.", "no, en realidad.", "puede ser."]
    return respuestas[hash(prompt) % 3] + continuacion[hash(prompt) % 3]


def main() -> int:
    print(chatbot_reglas("saludo", "Hola, como estas?"))
    print(chatbot_reglas("nombre", "Me llamo Carlos"))
    print(chatbot_reglas("despedida", "Chao, hasta luego"))
    # Retrieval
    faq = {
        "cual es el horario": "Lunes a viernes 9-18",
        "como contacto soporte": "soporte@empresa.com",
    }
    r, score = chatbot_retrieval("cual es el horario de atencion", faq)
    print(f"Retrieval: {r} (score={score:.2f})")
    # Generativo
    print(chatbot_generativo_mock("hola"))
    return 0


if __name__ == "__main__":
    sys.exit(main())