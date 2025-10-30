# Datasets

Esta carpeta contiene los conjuntos de datos utilizados a lo largo del diplomado.

## Organización

Los datasets están organizados por clase y tipo:

```
Datasets/
├── clase_01/          # Datasets de introducción
├── clase_02/          # Datasets para procesamiento
├── clase_03/          # Datasets para modelos estadísticos
├── clase_04/          # Datasets para deep learning
├── clase_05/          # Imágenes para CNN y segmentación
├── clase_06/          # Ambientes para RL
├── clase_07/          # Datasets para análisis de sesgos
├── clase_08/          # Datasets para MLOps
└── proyecto_final/    # Datasets para prueba técnica
```

## Fuentes de Datos

### Repositorios Recomendados
- **Kaggle**: https://www.kaggle.com/datasets
- **UCI ML Repository**: https://archive.ics.uci.edu/ml/
- **Google Dataset Search**: https://datasetsearch.research.google.com/
- **Datos Abiertos Colombia**: https://www.datos.gov.co/

### Datasets Clásicos
- **Iris**: Clasificación de flores
- **Boston Housing**: Predicción de precios
- **MNIST**: Dígitos escritos a mano
- **CIFAR-10**: Clasificación de imágenes
- **Titanic**: Análisis de supervivencia

## Notas Importantes

⚠️ **Los archivos de datos grandes no se suben al repositorio**

Por razones de tamaño, los datasets grandes se deben descargar por separado:

1. Consultar el README de cada clase para enlaces de descarga
2. Usar scripts de descarga automatizada cuando estén disponibles
3. Los datasets se ignoran en `.gitignore` (excepto ejemplos pequeños)

## Estructura de Archivos

Para mantener consistencia:
- Usar nombres descriptivos en snake_case
- Incluir metadatos (README.md) con cada dataset
- Documentar el origen y licencia de los datos
- Mantener versiones raw y procesadas separadas

## Versionamiento

Para datasets grandes, considerar usar **DVC** (Data Version Control):

```bash
dvc init
dvc add Datasets/mi_dataset.csv
git add Datasets/mi_dataset.csv.dvc
git commit -m "Add dataset"
```

## Licencias y Ética

⚖️ Siempre verificar:
- Licencia de uso del dataset
- Restricciones de redistribución
- Consideraciones de privacidad
- Citas apropiadas cuando se publiquen resultados
