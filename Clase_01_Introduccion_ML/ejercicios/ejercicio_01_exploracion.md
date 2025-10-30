# Ejercicio 1: Exploración de un Dataset
**Clase 1: Introducción a Machine Learning**

## Objetivos
- Practicar la carga y exploración de datos
- Aplicar análisis exploratorio de datos (EDA)
- Identificar características del dataset
- Preparar un reporte de exploración

## Descripción
En este ejercicio trabajarás con el dataset "Titanic" para realizar un análisis exploratorio completo.

## Instrucciones

### Parte 1: Cargar y Explorar (30 minutos)
1. Descarga el dataset de Titanic desde Kaggle o usa el de seaborn
2. Carga el dataset en un DataFrame de pandas
3. Responde las siguientes preguntas:
   - ¿Cuántas filas y columnas tiene el dataset?
   - ¿Qué tipos de datos contiene cada columna?
   - ¿Hay valores faltantes? ¿En qué columnas?

### Parte 2: Análisis Estadístico (30 minutos)
1. Calcula estadísticas descriptivas para variables numéricas
2. Encuentra:
   - La edad promedio de los pasajeros
   - El precio promedio de los tickets
   - La tasa de supervivencia general
3. Agrupa los datos por clase (Pclass) y analiza:
   - Tasa de supervivencia por clase
   - Precio promedio del ticket por clase

### Parte 3: Visualización (40 minutos)
Crea las siguientes visualizaciones:
1. Histograma de edades
2. Gráfico de barras de supervivencia por género
3. Box plot de tarifas por clase
4. Matriz de correlación de variables numéricas
5. Cualquier otra visualización que consideres interesante

### Parte 4: Insights (20 minutos)
Responde en un archivo markdown o notebook:
1. ¿Qué factores parecen influir más en la supervivencia?
2. ¿Hay algún patrón evidente en los datos?
3. ¿Qué problemas de calidad de datos identificaste?
4. ¿Qué pasos de preprocesamiento serían necesarios?

## Entregables
- [ ] Notebook de Jupyter con el análisis completo
- [ ] Código comentado y organizado
- [ ] Visualizaciones claras y etiquetadas
- [ ] Archivo README con conclusiones

## Recursos
- Dataset: `seaborn.load_dataset('titanic')` o Kaggle
- Documentación de pandas: https://pandas.pydata.org/
- Documentación de seaborn: https://seaborn.pydata.org/

## Evaluación
- Completitud del análisis: 30%
- Calidad del código: 25%
- Visualizaciones: 25%
- Insights y conclusiones: 20%

## Tips
- Usa las funciones de utils.py para facilitar el análisis
- No olvides manejar valores faltantes
- Comenta tu código para explicar tu razonamiento
- Incluye títulos y etiquetas en todas las gráficas

## Tiempo Estimado
2 horas

## Nivel de Dificultad
⭐⭐ Básico-Intermedio

---
**Fecha de entrega:** Ver calendario del diplomado
