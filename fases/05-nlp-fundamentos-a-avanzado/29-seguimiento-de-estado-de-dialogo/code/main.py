"""
Lección: 29-seguimiento-de-estado-de-dialogo
Fase: 05
Prerrequisitos: 28-evaluacion-de-contexto-largo
"""
from __future__ import annotations
import sys
import re
import numpy as np


class DialogueState:
    """Mock: maquina de estados para un dialogo multi-turno.
    Trackea slots: intent, slots, history.
    """
    def __init__(self):
        self.intent = None
        self.slots = {}
        self.history = []

    def update(self, utterance):
        """Actualiza el estado segun la utterance."""
        self.history.append(utterance)
        # Mock: detectar intent basico
        if re.search(r"\b(reservar|reserva|booking)\b", utterance, re.IGNORECASE):
            self.intent = "reservar"
        elif re.search(r"\b(cancelar|cancel)\b", utterance, re.IGNORECASE):
            self.intent = "cancelar"
        elif re.search(r"\b(hola|hi)\b", utterance, re.IGNORECASE):
            self.intent = "saludo"
        # Extraer slots
        for slot in ["fecha", "hora", "personas", "nombre"]:
            match = re.search(rf"{slot}\s*[:=]?\s*([\w\d\-:]+)", utterance, re.IGNORECASE)
            if match:
                self.slots[slot] = match.group(1)

    def is_complete(self):
        """Verifica si todos los slots requeridos estan llenos."""
        required = ["fecha", "hora", "personas", "nombre"]
        return all(s in self.slots for s in required)

    def missing_slots(self):
        required = ["fecha", "hora", "personas", "nombre"]
        return [s for s in required if s not in self.slots]

    def __repr__(self):
        return f"DialogueState(intent={self.intent}, slots={self.slots}, history={len(self.history)})"


def main() -> int:
    ds = DialogueState()
    ds.update("Hola, quiero reservar")
    ds.update("fecha: 2024-12-25")
    ds.update("hora: 19:00")
    ds.update("personas: 4")
    ds.update("nombre: Maria")
    print(f"Estado: {ds}")
    print(f"Completo: {ds.is_complete()}")
    print(f"Faltantes: {ds.missing_slots()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())