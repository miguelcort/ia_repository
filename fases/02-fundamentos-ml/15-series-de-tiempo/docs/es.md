# Series de tiempo

> Forecasting no es regresion: el tiempo tiene direccion y la autocorrelacion invalida el i.i.d. Sin TimeSeriesSplit, tu score vale cero.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-modelos-lineales-y-regresion-logistica
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar media movil.
- Implementar diferenciacion.
- Implementar autocorrelacion.
- Implementar forecast AR(1) recursivo.

## Constrúyelo

```python
def media_movil(x, ventana):
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(ventana - 1, n):
        out[i] = x[i - ventana + 1:i + 1].mean()
    return out
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-ts-pipeline
fase: 02
leccion: 15
---

1. Descomponer.
2. Test estacionariedad.
3. ARIMA -> SARIMA -> Prophet -> N-BEATS.
4. TimeSeriesSplit.
5. MAPE/RMSE/MASE.
```

## Ejercicios

1. **SARIMA**: implementa la estacionalidad (P,D,Q,m).
2. **Walk-forward**: implementa validacion rolling.
3. **Desafio**: implementa Prophet-lite con descomposicion +
   holidays.

## Lecturas recomendadas

- "Forecasting: Principles and Practice" (Hyndman & Athanasopoulos)
- statsmodels: <https://www.statsmodels.org/>
- Prophet: <https://facebook.github.io/prophet/>

---

> 📚 **Adaptación al español** de la lección "[Time Series]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).