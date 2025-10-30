# 🎓 Diplomado en Machine Learning
## Universidad Distrital Francisco José de Caldas

Repositorio oficial del módulo de Ciencia de Datos y Machine Learning. Este diplomado ofrece una formación completa en inteligencia artificial, desde fundamentos hasta implementación en producción.

---

## 📚 Estructura del Diplomado

El diplomado está organizado en **8 clases** que cubren todo el espectro del Machine Learning moderno:

### [Clase 1: Introducción a Machine Learning](./Clase_01_Introduccion_ML/)
Fundamentos de ML, tipos de aprendizaje, y flujo de trabajo en proyectos de ciencia de datos.
- 🎯 Conceptos básicos de IA y ML
- 🔍 Tipos de aprendizaje: supervisado, no supervisado, refuerzo
- 📊 Primeros modelos con Scikit-learn

### [Clase 2: Procesamiento de Datos](./Clase_02_Procesamiento_Datos/)
Técnicas de limpieza, transformación y preparación de datos para modelos de ML.
- 🧹 Limpieza y manejo de datos faltantes
- 📈 Análisis Exploratorio de Datos (EDA)
- ⚙️ Feature engineering y transformaciones

### [Clase 3: Modelos Estadísticos](./Clase_03_Modelos_Estadisticos/)
Modelos clásicos de machine learning y sus aplicaciones.
- 📉 Regresión lineal y logística
- 🌳 Árboles de decisión y Random Forest
- 🚀 Gradient Boosting (XGBoost, LightGBM)

### [Clase 4: Introducción a Deep Learning](./Clase_04_Introduccion_Deep_Learning/)
Fundamentos de redes neuronales y frameworks modernos.
- 🧠 Arquitectura de redes neuronales
- 🔄 Backpropagation y optimización
- 🛠️ TensorFlow y PyTorch

### [Clase 5: Convoluciones y Segmentación de Imágenes](./Clase_05_Convoluciones_Segmentacion/)
Redes neuronales convolucionales y procesamiento de imágenes.
- 🖼️ CNNs y arquitecturas clásicas
- 🎯 Clasificación y detección de objetos
- 🗺️ Segmentación semántica con U-Net

### [Clase 6: Aprendizaje por Refuerzo](./Clase_06_Aprendizaje_Refuerzo/)
Algoritmos de reinforcement learning y sus aplicaciones.
- 🎮 Q-Learning y algoritmos básicos
- 🤖 Deep Q-Networks (DQN)
- 🏆 Policy Gradients y Actor-Critic

### [Clase 7: Ética en Inteligencia Artificial](./Clase_07_Etica_IA/)
Consideraciones éticas, sesgos y IA responsable.
- ⚖️ Principios de ética en IA
- 🔍 Detección y mitigación de sesgos
- 🛡️ Privacidad y protección de datos
- 💡 Explicabilidad de modelos (XAI)

### [Clase 8: Docker y MLOps](./Clase_08_Docker_MLOps/)
Despliegue de modelos, contenedorización y ciclo de vida de ML en producción.
- 🐳 Docker para ML
- 🔄 Pipelines y automatización
- 🚀 Despliegue en producción
- 📊 Monitoreo de modelos

### [Prueba Técnica Final](./Prueba_Tecnica/)
Proyecto integrador que demuestra competencia en todo el ciclo de vida de ML.

---

## 📂 Organización del Repositorio

```
ia_repository/
│
├── Clase_01_Introduccion_ML/          # Fundamentos de ML
├── Clase_02_Procesamiento_Datos/      # Limpieza y preparación
├── Clase_03_Modelos_Estadisticos/     # Modelos clásicos
├── Clase_04_Introduccion_Deep_Learning/  # Redes neuronales
├── Clase_05_Convoluciones_Segmentacion/  # Visión computacional
├── Clase_06_Aprendizaje_Refuerzo/     # Reinforcement Learning
├── Clase_07_Etica_IA/                 # IA responsable
├── Clase_08_Docker_MLOps/             # Despliegue y producción
├── Prueba_Tecnica/                    # Proyecto final
├── Datasets/                          # Conjuntos de datos
└── Recursos/                          # Material adicional
```

Cada carpeta de clase contiene:
- 📖 `README.md`: Objetivos, contenido y tareas
- 📓 `notebooks/`: Jupyter notebooks con ejemplos
- 💻 `code/`: Scripts y utilidades
- 📝 `ejercicios/`: Ejercicios propuestos
- 📚 `referencias/`: Material de lectura

---

## 🚀 Cómo Usar Este Repositorio

### Prerrequisitos
- Python 3.8 o superior
- Git instalado
- Jupyter Notebook o JupyterLab
- (Opcional) Docker para la clase 8

### Configuración Inicial

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/miguelcort/ia_repository.git
   cd ia_repository
   ```

2. **Crear entorno virtual** (recomendado)
   ```bash
   python -m venv venv
   
   # En Windows
   venv\Scripts\activate
   
   # En Linux/Mac
   source venv/bin/activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Iniciar Jupyter**
   ```bash
   jupyter notebook
   ```

### Navegación
Cada clase es independiente pero progresiva. Se recomienda seguir el orden establecido para un mejor aprovechamiento.

---

## 🛠️ Tecnologías y Herramientas

### Lenguajes y Frameworks
- **Python**: Lenguaje principal
- **TensorFlow/Keras**: Deep Learning
- **PyTorch**: Deep Learning alternativo
- **Scikit-learn**: ML clásico

### Librerías de Datos
- **NumPy**: Computación numérica
- **Pandas**: Manipulación de datos
- **Matplotlib/Seaborn**: Visualización
- **Plotly**: Visualización interactiva

### MLOps y Despliegue
- **Docker**: Contenedorización
- **MLflow**: Tracking de experimentos
- **FastAPI**: APIs de modelos
- **DVC**: Versionamiento de datos

---

## 📖 Recursos Adicionales

### Documentación Oficial
- [Scikit-learn](https://scikit-learn.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [PyTorch](https://pytorch.org/)
- [Pandas](https://pandas.pydata.org/)

### Datasets
- [Kaggle](https://www.kaggle.com/datasets)
- [UCI ML Repository](https://archive.ics.uci.edu/ml/)
- [Google Dataset Search](https://datasetsearch.research.google.com/)
- [Datos Abiertos Colombia](https://www.datos.gov.co/)

### Comunidades
- [Stack Overflow](https://stackoverflow.com/questions/tagged/machine-learning)
- [Kaggle Forums](https://www.kaggle.com/discussion)
- [Reddit r/MachineLearning](https://www.reddit.com/r/MachineLearning/)

---

## 👥 Contribuciones

Este repositorio está en constante evolución. Las contribuciones son bienvenidas:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/mejora`)
3. Commit tus cambios (`git commit -m 'Agregar mejora'`)
4. Push a la rama (`git push origin feature/mejora`)
5. Abre un Pull Request

---

## 📝 Licencia

Este material es de uso académico para el diplomado de Machine Learning de la Universidad Distrital Francisco José de Caldas.

---

## 📧 Contacto

Para consultas sobre el diplomado:
- **Universidad Distrital Francisco José de Caldas**
- **Módulo**: Machine Learning y Ciencia de Datos

---

## ⭐ Agradecimientos

A todos los estudiantes y profesores que contribuyen al desarrollo de este diplomado.

**¡Éxitos en tu viaje por el mundo del Machine Learning! 🚀**
