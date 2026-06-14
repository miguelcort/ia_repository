# Fase 4 — Visión por computador

> De píxeles a comprensión: imagen, video, 3D, VLMs y modelos del mundo.

La visión por computador dejó de ser "clasificar ImageNet" hace
tiempo. Hoy cubre desde **detección y segmentación** en tiempo real
hasta **generación de imágenes y video con difusión**, pasando por
**visión 3D**, **transformers de visión**, y **modelos visión-
lenguaje** que combinan lo visual y lo textual. Esta fase recorre
todo ese espectro con la misma filosofía del currículo: **construir
antes de usar**. Empezamos con la convolución desde cero, llegamos
a ResNet implementado a mano, y luego subimos el nivel hasta usar
PyTorch y Hugging Face para Stable Diffusion, SAM, CLIP y ViT.

La fase tiene cuatro bloques. **Bloque 1 (lecciones 1–4)**: la base
discriminativa — píxeles, convoluciones, CNN clásicas, clasificación.
**Bloque 2 (5–8)**: localización — transfer learning, detección
(YOLO), segmentación semántica (U-Net) y de instancia (Mask R-CNN).
**Bloque 3 (9–11)**: generación — GANs, difusión clásica, Stable
Diffusion. **Bloque 4 (12–28)**: frontera — video, 3D, ViT,
open-vocabulary (CLIP, SAM), modelos visión-lenguaje, modelos del
mundo. El estudiante no tiene que dominarlas todas: la mayoría
terminará usando 5–8 de las 28 lecciones en su trabajo diario, y el
resto existe para dar contexto y poder leer papers recientes.

## Índice de lecciones

### Bloque 1 — Base discriminativa

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 01 | [Fundamentos de imagen](01-fundamentos-de-imagen/) | Aprender | Píxeles, canales, espacios de color, muestreo y filtros. |
| 02 | [Convoluciones desde cero](02-convoluciones-desde-cero/) | Construir | Kernel 2D, padding, stride y convolución como producto punto. |
| 03 | [CNNs: de LeNet a ResNet](03-cnns-desde-lenet-hasta-resnet/) | Construir | Arquitectura, *skip connections* y *bottleneck blocks*. |
| 04 | [Clasificación de imágenes](04-clasificacion-de-imagenes/) | Construir | Entrenamiento, augmentations y *learning rate finder*. |

### Bloque 2 — Localización y transfer learning

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 05 | [Transfer learning](05-transfer-learning/) | Construir | *Feature extraction* y *fine-tuning* con modelos preentrenados. |
| 06 | [Detección de objetos: YOLO](06-deteccion-de-objetos-yolo/) | Construir | *Anchor boxes*, IoU, NMS y arquitectura YOLOv8. |
| 07 | [Segmentación semántica: U-Net](07-segmentacion-semantica-unet/) | Construir | Encoder-decoder con *skip connections*. |
| 08 | [Segmentación de instancia: Mask R-CNN](08-segmentacion-de-instancia-mask-rcnn/) | Construir | Region proposals + máscara por instancia. |

### Bloque 3 — Generación

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 09 | [Generación de imágenes: GAN](09-generacion-de-imagenes-gan/) | Construir | Generator, discriminator y *loss* adversarial. |
| 10 | [Generación con difusión](10-generacion-de-imagenes-con-difusion/) | Construir | DDPM, *forward/reverse process* y *noise schedule*. |
| 11 | [Stable Diffusion](11-stable-diffusion/) | Construir | Latent diffusion, UNet, CLIP text encoder. |

### Bloque 4 — Frontera (video, 3D, ViT, VLM)

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 12 | [Comprensión de video](12-comprension-de-video/) | Construir | Modelado temporal, *3D CNN* y *video transformers*. |
| 13 | [Visión 3D: NeRFs](13-vision-3d-nerf/) | Construir | Neural radiance fields y renderizado volumétrico. |
| 14 | [Vision Transformers (ViT)](14-vision-transformers/) | Construir | *Patch embeddings*, atención y ViT desde cero. |
| 15 | [Visión en tiempo real en el borde](15-vision-en-tiempo-real-en-borde/) | Construir | TensorRT, ONNX, cuantización y Jetson. |
| 16 | [Pipeline de visión capstone](16-pipeline-de-vision-capstone/) | Construir | De captura a producción en una sola lección. |
| 17 | [Visión auto-supervisada](17-vision-auto-supervisada/) | Construir | SimCLR, DINO, MAE y contrastive learning. |
| 18 | [CLIP: vocabulario abierto](18-clip-vocabulario-abierto/) | Construir | Contrastive image-text pre-training. |
| 19 | [OCR y comprensión de documentos](19-ocr-y-comprension-de-documentos/) | Construir | Detección de texto, reconocimiento y layout. |
| 20 | [Recuperación de imágenes y *metric learning*](20-recuperacion-de-imagenes-y-metrica/) | Construir | Triplet loss, ArcFace, búsqueda por similitud. |
| 21 | [Puntos clave y estimación de poses](21-puntos-clave-y-estimacion-de-poses/) | Construir | Heatmaps, OpenPose, MediaPipe. |
| 22 | [3D Gaussian Splatting desde cero](22-gaussian-splatting-3d/) | Construir | Representación explícita + rasterización. |
| 23 | [Diffusion transformers y *rectified flow*](23-difusion-transformers-y-flujo-rectificado/) | Construir | DiT, SiT y transporte óptimo. |
| 24 | [SAM 3 y segmentación open-vocabulary](24-sam3-segmentacion-de-vocabulario-abierto/) | Construir | *Promptable segmentation* con texto y puntos. |
| 25 | [Modelos visión-lenguaje](25-modelos-de-vision-lenguaje/) | Construir | ViT-MLP-LLM y arquitecturas tipo LLaVA. |
| 26 | [Profundidad monocular y geometría](26-profundidad-monocular/) | Construir | DPT, MiDaS y estructura a partir de motion. |
| 27 | [Seguimiento de múltiples objetos](27-seguimiento-de-multiples-objetos/) | Construir | SORT, ByteTrack y memoria de video. |
| 28 | [Modelos del mundo y difusión de video](28-modelos-del-mundo-y-difusion-de-video/) | Construir | World models, Sora, Wan y consistencia temporal. |

## Prerrequisitos

- **Fases 0, 1, 2 y 3** completas.
- Conocimiento de álgebra lineal (convolución como producto punto).
- Conceptos básicos de probabilidad (cross-entropy, softmax).
- GPU recomendada (CUDA o MPS); muchas lecciones funcionan en
  CPU con datasets pequeños.

## Objetivos de la fase

Al terminar esta fase el estudiante podrá:

- **Implementar** convolución, pooling y CNN desde cero y razonar
  sobre campos receptivos.
- **Construir** un pipeline de detección y segmentación con
  PyTorch, desde el dataset hasta el *inference*.
- **Comprender** GANs, VAEs, difusión clásica y Stable Diffusion
  sin tratar a la librería como caja negra.
- **Aplicar** transfer learning a un problema nuevo en menos de
  un día de trabajo.
- **Desplegar** un modelo de visión en *edge* (Jetson, ONNX
  Runtime) o servidor (TorchServe, Triton).
- **Leer** papers recientes sobre ViT, SAM, modelos del mundo y
  *diffusion transformers*.

## Stack y herramientas

- **PyTorch** y **torchvision** como núcleo.
- **OpenCV** y **Pillow** para I/O de imagen.
- **Albumentations** para aumentación.
- **Hugging Face Transformers / Diffusers** para modelos
  preentrenados.
- **PyTorch Lightning** opcional para entrenamiento.
- **Detectron2 / Ultralytics** para detección y segmentación.
- **TensorRT / ONNX Runtime** para despliegue en el borde.
- **wandb** o **TensorBoard** para tracking.

## Conceptos clave

| Concepto | Aparece en | Reaparece en |
|---|---|---|
| **Convolución** | Lección 02 | Fase 6 (audio), Fase 12 (multimodal) |
| **ResNet** | Lección 03 | Backbone de todas las CNNs modernas |
| **YOLO** | Lección 06 | Aplicaciones en tiempo real |
| **U-Net** | Lección 07 | Stable Diffusion, segmentación médica |
| **Difusión** | Lecciones 10, 11, 23 | Fase 8 (generativa), Fase 12 (video) |
| **CLIP** | Lección 18, 11 | Fase 12 (multimodal), Fase 14 (agentes visuales) |
| **ViT** | Lección 14 | Fase 7 (transformers), Fase 12 |
| **NeRF / 3DGS** | Lecciones 13, 22 | Reconstrucción 3D |
| **VLM** | Lección 25 | Fase 12, Fase 14 |

## Cómo estudiar esta fase

1. **Resuelve los bloques 1 y 2 antes de saltar al 3.** Sin
   entender la convolución y la segmentación, la generación se
   vuelve misteriosa.
2. **Usa datasets pequeños al principio** (MNIST, Fashion-MNIST,
   CIFAR-10). Saltar directo a ImageNet frustra y no enseña.
3. **Compara siempre con la librería oficial.** Si implementas
   ResNet y tu accuracy en CIFAR-10 está a 3% de `torchvision`,
   vas bien.
4. **Las lecciones 11, 18, 22 y 25 son las más "famosas"** — si
   tu tiempo es limitado, enfócate ahí.
5. **Lección 16 es integradora.** Úsala como proyecto parcial:
   un pipeline de detección de defectos en una línea de
   producción simulada.

## Verificación de progreso

```bash
# Lección 02 — convolución desde cero
python3 fases/04-vision-por-computador/02-convoluciones-desde-cero/code/main.py

# Lección 03 — ResNet en CIFAR-10
python3 fases/04-vision-por-computador/03-cnns-desde-lenet-hasta-resnet/code/main.py

# Lección 11 — Stable Diffusion con diffusers
python3 fases/04-vision-por-computador/11-stable-diffusion/code/main.py
```

Si los tres demos terminan con código 0 y producen las salidas
esperadas, la fase está aprobada.

## Cuándo usar cada familia

| Problema | Familia | Lección |
|---|---|---|
| Clasificación de imágenes | CNN / ViT | 03, 14 |
| Detección en tiempo real | YOLO | 06 |
| Segmentación por clase | U-Net | 07 |
| Contar objetos individuales | Mask R-CNN | 08 |
| Generar imágenes | Difusión | 10, 11 |
| Buscar imágenes por texto | CLIP | 18 |
| Reconstrucción 3D | NeRF / 3DGS | 13, 22 |
| Captioning de imágenes | VLM | 25 |

## Conexión con otras fases

- **Entrada** → [Fase 3 — Núcleo de Deep Learning](../03-nucleo-deep-learning/README.md).
- **Salida natural** → [Fase 5 — NLP](../05-nlp-fundamentos-a-avanzado/README.md)
  (la arquitectura transformer de la Fase 7 nace de ViT).
- **Reuso en** → Fase 8 (generativa), Fase 12 (multimodal),
  Fase 14 (agentes con visión), Fase 15 (sistemas autónomos).

## Recursos recomendados

- *Computer Vision: Algorithms and Applications* — Szeliski (PDF libre).
- *Deep Learning for Computer Vision* — Rosebrock.
- *Programming Computer Vision with Python* — Solem.
- *ultralytics* docs — <https://docs.ultralytics.com>.
- *Hugging Face Diffusers* — <https://huggingface.co/docs/diffusers>.

## Véase también

- [glosario/terminos.md](../../glosario/terminos.md) — *convolution*,
  *IoU*, *NMS*, *CLIP*, *diffusion*.
- [Fase 3 — Núcleo de Deep Learning](../03-nucleo-deep-learning/README.md).
- [Fase 7 — Transformers a fondo](../07-transformers-a-fondo/README.md).
- [ROADMAP.md](../../ROADMAP.md) — estado de las 20 fases.

---

> 📚 **Adaptación al español** del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> de Rohit Ghumare (MIT). Ver [CREDITS.md](../../CREDITS.md).
