# Seguimiento de estado de diálogo

> Mantener el estado del dialogo multi-turno: intent, slots, historial. Datasets: MultiWoZ, SGD. SOTA: BERT/SOM-DST (90%+ JGA) o LLM zero-shot. Frameworks: Rasa, simple-dst, DeepPavlov. Aplicaciones: task-oriented dialogs (booking, banking, support).

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 28-evaluacion-de-contexto-largo
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar maquina de estados de dialogo.
- Detectar intent y slots.
- Verificar completitud.
- Diagnosticar DST clasico vs LLM-based.

## Constrúyelo

```python
class DialogueState:
    def update(self, utterance):
        self.history.append(utterance)
        if re.search(r"\b(reservar|reserva)\b", utterance, re.IGNORECASE):
            self.intent = "reservar"
        for slot in ["fecha", "hora", "personas", "nombre"]:
            match = re.search(rf"{slot}\s*[:=]?\s*([\w\d]+)", utterance, re.IGNORECASE)
            if match:
                self.slots[slot] = match.group(1)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-dst
fase: 05
leccion: 29
---

1. Default: BERT fine-tune MultiWoZ/SGD (90%+ JGA).
2. LLM zero-shot: Claude/GPT con schema prompt.
3. Rasa: rules + ML, dominios pequenos.
4. Hibrido: BERT + LLM fallback.
5. Multi-domain: train por separado o shared.
6. JGA eval, simple-dst framework.
```

## Ejercicios

1. **DST BERT**: fine-tunear BERT en MultiWoZ, JGA > 80.
2. **LLM zero-shot DST**: implementar DST con Claude
   zero-shot.
3. **Desafio**: sistema completo de task-oriented dialog
   con DST, policy, NLG. Rasa o simple-dst. Eval JGA en
   golden set.

## Lecturas recomendadas

- "MultiWoZ" (Budzianowski et al., 2018)
- "SGD" (Rastogi et al., 2020)
- "simple-dst": <https://github.com/laituan245/simple-dst>

---

> 📚 **Adaptación al español** de la lección "[Dialogue State Tracking]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).