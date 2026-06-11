# StyleGAN

> StyleGAN (Karras 2019-2021): mapping z→w, AdaIN en cada layer con w, style mixing, noise injection, progressive growing, truncation trick. StyleGAN2: weight modulation + demodulation. StyleGAN3: alias-free. SOTA para face generation. Aplicaciones: style transfer, inversion, editing, super-resolution. Limitaciones: dataset bias, training costoso, no 3D.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 08/03-gans-generador-y-discriminador
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar mapping network z → w.
- Implementar AdaIN.
- Aplicar style mixing y truncation.
- Diagnosticar trade-offs de progressive growing.

## Constrúyelo

```python
def adain(x, w_y, w_b):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return w_y * (x - mu) / np.sqrt(var + 1e-6) + w_b
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-stylegan
fase: 08
leccion: 05
---

1. z -> w mapping, AdaIN, style mixing.
2. Truncation: w_avg + psi*(w - w_avg).
3. StyleGAN2: weight modulation.
4. StyleGAN3: alias-free.
5. Aplicaciones: faces, edit, inversion.
```

## Ejercicios

1. **Inversion**: implementar encoder
   (e4e, pSp) para invertir imagen real.
2. **Editing**: implementar InterfaceGAN
   para editar atributos.
3. **Desafio**: entrenar StyleGAN3
   en custom dataset.

## Lecturas recomendadas

- "A Style-Based Generator Architecture for Generative Adversarial Networks" (Karras et al., 2019)
- "Analyzing and Improving the Image Dynamics of StyleGAN" (Karras et al., 2020)
- "Alias-Free Generative Adversarial Networks" (Karras et al., 2021)

---

> 📚 **Adaptación al español** de la lección "[StyleGAN]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).