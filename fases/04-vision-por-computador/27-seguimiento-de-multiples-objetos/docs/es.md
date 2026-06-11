# Seguimiento de multiples objetos

> Asignar ID consistente a cada objeto a lo largo del tiempo. SORT, DeepSORT, ByteTrack, BoT-SORT. Base de analytics deportivo, retail, ADAS, robotics.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 06-deteccion-de-objetos-yolo
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Calcular IoU y matriz de costos.
- Implementar asignacion hungara.
- Implementar Kalman filter basico.
- Diagnosticar SORT/ByteTrack/BoT-SORT.

## Constrúyelo

```python
def hungarian_assignment(cost_matrix):
    N, M = cost_matrix.shape
    pares = []
    disponibles = list(range(M))
    for i in range(N):
        if not disponibles:
            break
        costos = sorted([(cost_matrix[i, j], j) for j in disponibles])
        if costos[0][0] < 1.0:
            j = costos[0][1]
            pares.append((i, j))
            disponibles.remove(j)
    return pares
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-mot
fase: 04
leccion: 27
---

1. Default: YOLOv8 + ByteTrack.
2. Crowded: BoT-SORT.
3. Oclusion: DeepSORT.
4. Video segmentation: SAM 2, MaskTrack R-CNN.
5. MOTA + IDF1 metricas.
```

## Ejercicios

1. **SORT completo**: Kalman + IoU + Hungarian + ID
   management.
2. **ByteTrack**: asociar low-score detections tambien.
3. **Desafio**: tracking de jugadores en un partido de
   futbol (SportsMOT dataset) con BoT-SORT.

## Lecturas recomendadas

- "SORT" (Bewley et al., 2016)
- "ByteTrack" (Zhang et al., 2022)
- "BoT-SORT" (Aharon et al., 2022)
- boxmot: <https://github.com/mikel-brostrom/boxmot>

---

> 📚 **Adaptación al español** de la lección "[Multi-Object Tracking]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).