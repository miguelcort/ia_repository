# Puntos clave y estimación de poses

> Localizar partes del cuerpo, manos, cara. COCO 17 keypoints, MediaPipe 33 (body), 21 (manos), 468 (cara). SOTA: ViTPose, RTMPose, DWPose.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 06-deteccion-de-objetos-yolo
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Detectar keypoints (mock).
- Construir conexiones del esqueleto.
- Calcular angulos articulares.
- Evaluar con PCK y OKS mAP.

## Constrúyelo

```python
def angulo_articulacion(a, b, c):
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cos = float(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-pose-elegir
fase: 04
leccion: 21
---

1. Mobile: MediaPipe Pose.
2. Cloud max: ViTPose, RTMPose.
3. Multi-person: top-down YOLO+ViTPose, bottom-up DWPose.
4. Manos: MediaPipe Hands (21).
5. Cara: MediaPipe Face Mesh (468).
```

## Ejercicios

1. **Skeleton drawing**: dibujar conexiones entre keypoints
   en una imagen.
2. **Action recognition from pose**: clasificar acciones
   (caminar, correr) con features geometricos.
3. **Desafio**: integrar MediaPipe Pose en una app de
   conteo de repeticiones de sentadillas.

## Lecturas recomendadas

- "OpenPose" (Cao et al., 2017)
- "HRNet" (Sun et al., 2019)
- "ViTPose" (Xu et al., 2022)
- MediaPipe: <https://mediapipe.dev/>

---

> 📚 **Adaptación al español** de la lección "[Keypoints and Pose Estimation]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).