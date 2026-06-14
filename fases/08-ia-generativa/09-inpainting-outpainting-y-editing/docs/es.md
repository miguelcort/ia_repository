# 09 — Inpainting, outpainting y edición de imagen

> Text-to-image crea cosas nuevas. Inpainting arregla las viejas. En producción, el 70% del trabajo facturable de imagen es edición — cambiar un fondo, eliminar un logo, extender el lienzo, regenerar una mano. Inpainting es donde la difusión gana su lugar.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 8 · 07 (Difusión latente), Fase 8 · 08 (ControlNet y LoRA)
**Tiempo estimado:** ~75 minutos

## Objetivos de aprendizaje

- Explicar la diferencia entre inpainting, outpainting y edición
  de imagen con difusión.
- Implementar un inpainter DDPM simplificado en datos 1-D/5-D
  con reinjection de la región no enmascarada.
- Comparar SDEdit, InstructPix2Pix, RePaint y modelos de
  inpainting propios (9 canales).
- Diagnosticar las trampas más comunes: *seams*,
  *mask leakage*, interacción con CFG.
- Combinar SAM con un inpainter para pipelines de
  eliminación/cambio de fondo.

## El problema

Un cliente envía una foto de producto perfecta con un letrero
distractivo de fondo. Quieres borrar el letrero y dejar todo lo
demás pixel-idéntico. No puedes correr text-to-image desde cero
— el resultado tendrá un color diferente, iluminación diferente,
ángulo de producto diferente. Quieres regenerar **solo** la
región enmascarada, y quieres que la regeneración respete el
contexto circundante.

Eso es inpainting. Variantes:

- **Inpainting.** Regenerar dentro de una máscara, mantener los
  píxeles de afuera.
- **Outpainting.** Regenerar fuera de una máscara (o más allá
  del lienzo), mantener adentro.
- **Edición de imagen.** Regenerar toda la imagen pero mantener
  fidelidad semántica o estructural al original (SDEdit,
  InstructPix2Pix).

Cada *pipeline* de difusión en 2026 envía un modo de
inpainting. Flux.1-Fill, Stable Diffusion Inpaint, SDXL-Inpaint,
DALL-E 3 Edit. Funcionan sobre el mismo principio.

## El concepto

### El enfoque ingenuo (y por qué está mal)

Corre text-to-image estándar con una máscara. En cada paso de
muestreo, reemplaza la región no enmascarada del latente ruidoso
con la imagen limpia forward-diffusionada. Funciona... mal.
Artefactos de borde se filtran porque el modelo no tiene
información sobre lo que está en la región enmascarada.

### El modelo de inpainting propiamente dicho

Entrena un U-Net modificado que toma 9 canales de entrada en
vez de 4:

```text
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

Los canales extra son una copia de la imagen fuente codificada
por el VAE más una máscara de un solo canal. En entrenamiento,
enmascaras regiones al azar y entrenas al modelo a *denoisear*
solo la región enmascarada mientras la región no enmascarada se
da como señal limpia de condicionamiento. En inferencia, el
modelo puede "ver" lo que rodea la región enmascarada y produce
completions coherentes.

SD-Inpaint, SDXL-Inpaint, Flux-Fill usan esta entrada de 9
canales (o análoga). Diffusers `StableDiffusionInpaintPipeline`,
`FluxFillPipeline`.

### SDEdit (Meng et al., 2022) — edición libre

Añade ruido a la imagen fuente hasta algún `t` intermedio, luego
corre la cadena reversa desde `t` hasta 0 con un nuevo prompt.
Sin reentrenamiento. La elección de `t` inicial intercambia
fidelidad por libertad creativa:

- `t/T = 0.3` → casi idéntico a la fuente, cambios estilísticos
  pequeños.
- `t/T = 0.6` → ediciones moderadas, preserva estructura
  gruesa.
- `t/T = 0.9` → generado casi desde ruido, mínima preservación
  de la fuente.

### InstructPix2Pix (Brooks et al., 2023)

*Ajusta* un modelo de difusión sobre triples
`(input_image, instruction, output_image)`. En inferencia, condiciona
sobre la imagen de entrada y una instrucción de texto
("hazlo de atardecer", "añade un dragón"). Dos escalas CFG:
escala de imagen y escala de texto.

### RePaint (Lugmayr et al., 2022)

Mantén un modelo de difusión no-condicional estándar. En cada
paso reverso, *resample* — salta a un estado más ruidoso
ocasionalmente y regenera. Evita artefactos de borde. Se usa
cuando no tienes un modelo de inpainting entrenado.

## Constrúyelo

`code/main.py` implementa un esquema de inpainting 1-D de
juguete sobre datos de 5 dimensiones. Entrenamos un DDPM sobre
datos de mezcla 5-D donde cada muestra es 5 floats de uno de dos
clústeres. En inferencia, "enmascaramos" 2 de las 5 dimensiones,
inyectamos la versión forward-noisy de las tres no enmascaradas
en cada paso, y regeneramos solo las dimensiones enmascaradas.

### Paso 1: datos 5-D del DDPM

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### Paso 2: entrenar denoiser sobre las 5 dimensiones

DDPM estándar. La red produce predicción de ruido 5-D para la
entrada ruidosa 5-D.

### Paso 3: en inferencia, reverso consciente de máscara

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + \
                math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...luego corre el paso reverso normal sobre x_t
```

Este es el enfoque ingenuo y funciona en datos 1-D de juguete.
El inpainting real de imagen usa la entrada de 9 canales porque
la coherencia de textura importa más.

### Paso 4: outpainting

Outpainting es inpainting con la máscara invertida: enmascara el
nuevo lienzo (previamente inexistente), llena el resto con el
original. Objetivo de entrenamiento idéntico.

## Trampas comunes

- **Costuras (seams).** El enfoque ingenuo deja bordes
  visibles porque la información de gradiente no fluye a través
  de la máscara. Fix: dilata la máscara 8–16 píxeles, o usa un
  modelo de inpainting propiamente dicho.
- **Filtración de máscara.** Si la región no enmascarada de la
  imagen de condicionamiento es de baja calidad o ruidosa,
  contamina la generación dentro de la máscara. Denoisa o
  desenfoca ligeramente.
- **CFG interactúa con el tamaño de la máscara.** CFG alto
  sobre una máscara pequeña = parche saturado. Reduce CFG para
  ediciones pequeñas.
- **Acantilado de fidelidad de SDEdit.** Ir de `t/T = 0.5` a
  `t/T = 0.6` puede perder la identidad del sujeto. Barrido y
  *checkpoint*.
- **Mismatch de prompt.** El prompt debe describir la imagen
  *completa*, no solo el nuevo contenido. "Un gato sentado en
  una silla", no "un gato".

## Úsalo

| Tarea | Pipeline |
|---|---|
| Eliminar objeto, máscara pequeña | SD-Inpaint o Flux-Fill, prompt estándar. |
| Reemplazar cielo | SD-Inpaint + "cielo azul al atardecer". |
| Extender lienzo | SDXL modo outpaint (8px *feather*) o Flux-Fill con máscara de outpaint. |
| Regenerar mano/cara | SD-Inpaint con prompt que re-describe al sujeto + ControlNet-Openpose. |
| Cambiar estilo de una región | SDEdit a `t/T=0.5` sobre la región enmascarada. |
| "Hazlo de atardecer" | InstructPix2Pix o Flux-Kontext. |
| Reemplazo de fondo | Máscara SAM → SD-Inpaint. |
| Ultra-alta fidelidad | Flux-Fill o GPT-Image (hosteado) para los casos más difíciles. |

SAM (Segment Anything de Meta, 2023) + diffusion inpaint es el
*pipeline* de eliminación de fondo de 2026. SAM 2 (2024)
funciona sobre video.

## Despliégalo

Guarda `outputs/skill-editing-pipeline.md`. El *skill* toma una
imagen original + descripción de edición + máscara opcional (o
prompt de SAM) y devuelve: enfoque de generación de máscara,
modelo base, escalas CFG (imagen + texto), SDEdit-t o modo
inpainting, y checklist de QA.

## Ejercicios

1. **Fácil.** En `code/main.py`, varía la fracción de
   dimensiones enmascaradas de 0.2 a 0.8. ¿En qué fracción la
   calidad de inpainting (residual en dimensiones enmascaradas)
   iguala a la generación no condicional?
2. **Medio.** Implementa RePaint: cada 10 pasos reversos,
   salta 5 pasos hacia atrás (añade ruido) y re-denoisa.
   Mide si reduce el residual de borde en el borde de la
   máscara.
3. **Difícil.** Usa Hugging Face diffusers para comparar:
   SD 1.5 Inpaint + ControlNet-Openpose vs Flux.1-Fill sobre
   20 tareas de regeneración de cara. Puntúa adherencia a
   pose y preservación de identidad por separado.

## Términos clave

| Término | Lo que dice la gente | Lo que realmente significa |
|---|---|---|
| **Inpainting** | "Rellena el hueco" | Regenerar dentro de una máscara; mantener los píxeles de afuera. |
| **Outpainting** | "Extiende el lienzo" | Regenerar fuera del lienzo; mantener adentro. |
| **U-Net de 9 canales** | "Modelo de inpainting propiamente dicho" | U-Net con `noisy \| encoded-source \| mask` como entrada. |
| **SDEdit** | "Img2img con nivel de ruido" | Ruido hasta tiempo `t`, *denoise* con nuevo prompt. |
| **InstructPix2Pix** | "Ediciones solo con texto" | Difusión ajustada sobre triples (imagen, instrucción, salida). |
| **RePaint** | "Sin reentrenamiento" | Re-ruido periódico durante el reverso para reducir costuras. |
| **SAM** | "Segment Anything" | Generador de máscaras por clics o *boxes*; se combina con inpaint. |
| **Flux-Kontext** | "Edita con contexto" | Variante de Flux que acepta imagen de referencia + instrucción para ediciones. |

## Nota de producción: los *pipelines* de edición son sensibles a latencia

Los usuarios que editan una imagen esperan round-trips sub-5
segundos. Un SDXL-Inpaint de 30 pasos a 1024² es 3-4 s en una
L4, más la generación de máscara SAM (~200 ms) y encode/decode
VAE (~500 ms combinados). En *framing* de producción, esto está
*bound* por TTFT más que por *throughput* — batch 1, baja
concurrencia, minimiza cada etapa:

- **SAM-H es el lento.** SAM-H a 1024² es ~200 ms; SAM-ViT-B
  es ~40 ms con pérdida menor de calidad. SAM 2 (video) añade
  *overhead* temporal; no lo uses para ediciones de imagen
  única.
- **Sáltate el encode cuando sea posible.**
  `pipe.image_processor.preprocess(img)` codifica a latentes.
  Si tienes los latentes de la generación anterior (típico en
  UIs de edición iterativa), pásalos directamente vía
  `latents=...` para saltarte un encode VAE.
- **La dilatación de máscara importa para el *throughput*
  también.** Una máscara pequeña significa que la mayor parte
  del forward pass del U-Net se desperdicia (los píxeles no
  enmascarados se *clampan* de todos modos). El
  `StableDiffusionInpaintPipeline` de `diffusers` corre el
  U-Net completo independientemente; solo las variantes
  *proper-inpaint* de 9 canales explotan cómputo enmascarado.
- **Flux-Kontext es la respuesta 2025.** Un solo forward pass
  sobre `(source_image, instruction)` — sin máscara separada,
  sin barrido de ruido de SDEdit. En una H100 envía una edición
  en ~1.5 s. La lección arquitectónica: colapsa las etapas.

## Lecturas recomendadas

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — inpainting sin entrenamiento.
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit.
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — edición por instrucción de texto.
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM, la fuente de máscaras.
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — SAM para video.
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — edición a nivel de atención.
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — *tooling* 2024.

---

> 📚 **Adaptación al español** de la lección
> "[Inpainting, Outpainting & Image Editing]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
