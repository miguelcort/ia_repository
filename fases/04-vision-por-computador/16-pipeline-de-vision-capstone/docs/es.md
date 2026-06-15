# 16 — Pipeline de visión capstone

> Un pipeline de visión de extremo a extremo: captura → preprocesamiento → inferencia → post-procesamiento → acción. El capstone de la fase 4.

**Tipo:** Construir
**Lenguajes:** Python, C++ opcional
**Prerrequisitos:** Todas las lecciones anteriores de fase 4
**Tiempo estimado:** ~50 horas

## Objetivos de aprendizaje

- Construir un pipeline de visión de extremo a extremo
  para un problema real.
- Combinar detección, segmentación, y tracking.
- Desplegar con APIs, GPU, edge.
- Medir latencia y throughput end-to-end.

## El problema

El capstone de la Fase 4 te pide construir un sistema
completo. Ejemplos:

- **Inspección industrial:** detectar defectos en
  productos manufacturados con cámara.
- **Conteo de inventario:** contar items en estantes
  desde fotos.
- **Análisis deportivo:** tracking de jugadores en un
  partido.
- **Tráfico:** detección de incidentes en cámaras de
  carretera.

Cada uno requiere la integración de varias lecciones:
captura de video, preprocesamiento, modelo de detección,
post-procesamiento, alertas o visualización. La lección
cubre el patrón general; el proyecto específico lo eliges
tú.

## El concepto

**Componentes típicos de un pipeline de visión.**

1. **Captura:** cámara, video file, RTSP stream.
2. **Preprocesamiento:** resize, normalización, color
   conversion.
3. **Modelo de inferencia:** detección (YOLO),
   segmentación (U-Net), clasificación (ViT).
4. **Post-procesamiento:** NMS, thresholding, tracking
   (ByteTrack, DeepSORT).
5. **Lógica de negocio:** alertas, agregación,
   persistencia.
6. **Visualización:** dibujar bounding boxes, máscaras.

**Latencia end-to-end.** Cada componente añade latencia.
Para tiempo real, el total debe ser < 33ms (30 FPS).
Componentes típicos:

- Captura: 1-5 ms.
- Preprocesamiento: 1-2 ms.
- Inferencia: 5-20 ms (con TensorRT).
- Post-procesamiento: 1-5 ms.
- Total: 10-30 ms.

**Patrón de deployment.**

```python
class Pipeline:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def process_frame(self, frame):
        # 1. Preprocess
        tensor = self.preprocess(frame)
        # 2. Inference
        outputs = self.model(tensor)
        # 3. Postprocess
        results = self.postprocess(outputs)
        return results

    def run(self, video_source):
        for frame in video_source:
            results = self.process_frame(frame)
            self.handle_results(results)
```

**Métricas a monitorear.**

- **Latencia P50/P95/P99:** no solo el promedio. P99
  revela outliers.
- **Throughput:** FPS en condiciones reales.
- **Accuracy por clases:** algunas clases se degradan
  más con cambios en iluminación.
- **Drift detection:** la distribución de detecciones
  cambia con el tiempo; alerta.

**Trampas.**

- **No medir en producción:** "funciona en mi laptop" no
  es generalizable. Mide en el hardware target.
- **Ignorar la cámara:** la calidad de la imagen
  determina la accuracy. Invierte en buena óptica.
- **Sin logging:** no saber qué falló. Loguea cada
  predicción con su score.

## Constrúyelo

```python
import numpy as np


class VisionPipeline:
    """Pipeline mínimo de detección en video."""

    def __init__(self, model, class_names, conf_threshold=0.5,
                 iou_threshold=0.5):
        self.model = model
        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def preprocess(self, frame, target_size=640):
        """Resize y normalización."""
        h, w = frame.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        # Resize (simulado)
        frame = frame[:new_h, :new_w, :]
        # Pad a cuadrado
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)))
        # Normalizar
        return frame.astype(np.float32) / 255.0

    def postprocess(self, outputs):
        """Filtrar por confidence y aplicar NMS."""
        # Implementación simplificada
        detections = []
        for det in outputs:
            if det["conf"] >= self.conf_threshold:
                detections.append(det)
        return detections

    def draw(self, frame, detections):
        """Dibujar bounding boxes en el frame."""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = f"{self.class_names[det['class']]}: {det['conf']:.2f}"
            # Aquí iría cv2.rectangle y cv2.putText
        return annotated

    def process_frame(self, frame):
        tensor = self.preprocess(frame)
        outputs = self.model(tensor)
        detections = self.postprocess(outputs)
        return self.draw(frame, detections)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-vision-pipeline
fase: 04
leccion: 16
---

Eres un asistente que diseña pipelines de visión para
producción. Recibirás la tarea, el hardware, y la
latencia objetivo. Tu trabajo:

1. Captura: gstreamer para Linux, AVFoundation para
   macOS, DirectShow para Windows.
2. Preprocesamiento: cv2.resize + normalización
   ImageNet.
3. Modelo: YOLOv8n para detección, ViT-S para
   clasificación, U-Net para segmentación.
4. Inferencia: TensorRT para GPU NVIDIA, TFLite para
   edge.
5. Post-procesamiento: NMS con threshold 0.5.
6. Tracking: ByteTrack para multi-object tracking.
7. Logging: cada detección con timestamp, score, clase.
8. Métricas: latencia P50/P95/P99, throughput, accuracy
   por clase.
9. Deployment: API REST con FastAPI, o gRPC para
   latencia baja.
```

## Ejercicios

1. **Pipeline mínimo**: implementa un pipeline de
   detección con webcam y dibuja las bounding boxes.
2. **Latencia**: mide latencia P50, P95, P99 sobre 1000
   frames.
3. **Desafío**: implementa tracking con ByteTrack o
   DeepSORT.

## Lecturas recomendadas

- *Designing Machine Learning Systems* — Huyen.
- *Computer Vision: Algorithms and Applications* —
  Szeliski.
- ByteTrack: <https://github.com/ifzhang/ByteTrack>.
- FastAPI: <https://fastapi.tiangolo.com>.
- GStreamer: <https://gstreamer.freedesktop.org>.

---

> 📚 **Adaptación al español** de la lección "[Vision Pipeline Capstone]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
