# 🚀 Guía de Inicio Rápido

Bienvenido al Diplomado en Machine Learning de la Universidad Distrital. Esta guía te ayudará a empezar rápidamente.

## 📋 Antes de Empezar

### Requisitos Previos
- Python 3.8 o superior instalado
- Conocimientos básicos de Python
- Git instalado en tu sistema
- 10GB de espacio libre en disco

### Conocimientos Recomendados
- Programación en Python (básico)
- Matemáticas: álgebra lineal básica, estadística
- Manejo de línea de comandos

## ⚡ Instalación Rápida

### 1. Clonar el Repositorio
```bash
git clone https://github.com/miguelcort/ia_repository.git
cd ia_repository
```

### 2. Crear Entorno Virtual
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verificar Instalación
```python
python -c "import numpy, pandas, sklearn, tensorflow; print('¡Todo listo!')"
```

### 5. Iniciar Jupyter
```bash
jupyter lab
```

## 📚 Primeros Pasos

### Día 1: Familiarízate con la Estructura
1. Lee el [README principal](./README.md)
2. Explora la estructura de carpetas
3. Revisa el [README de la Clase 1](./Clase_01_Introduccion_ML/README.md)

### Día 2: Primer Notebook
1. Abre `Clase_01_Introduccion_ML/notebooks/01_primer_modelo.ipynb`
2. Ejecuta celda por celda
3. Experimenta modificando el código

### Día 3: Primer Ejercicio
1. Lee `Clase_01_Introduccion_ML/ejercicios/ejercicio_01_exploracion.md`
2. Intenta resolverlo por tu cuenta
3. Compara con las soluciones (cuando estén disponibles)

## 🛠️ Herramientas Recomendadas

### Editores de Código
- **Jupyter Lab** (ya incluido) - Para notebooks
- **VS Code** - Editor versátil con extensiones de Python
- **PyCharm Community** - IDE completo para Python

### Extensiones Útiles para VS Code
- Python
- Jupyter
- Pylance
- Python Indent
- GitLens

## 📖 Ruta de Aprendizaje Sugerida

### Semanas 1-2: Fundamentos
- ✅ Clase 1: Introducción a ML
- ✅ Clase 2: Procesamiento de Datos
- 📝 Ejercicios 1-3

### Semanas 3-4: Modelos Clásicos
- ✅ Clase 3: Modelos Estadísticos
- 📝 Ejercicios 4-6
- 🎯 Mini-proyecto: Predicción de precios

### Semanas 5-6: Deep Learning
- ✅ Clase 4: Introducción a Deep Learning
- ✅ Clase 5: Convoluciones y Segmentación
- 📝 Ejercicios 7-10

### Semanas 7-8: Temas Avanzados
- ✅ Clase 6: Aprendizaje por Refuerzo
- ✅ Clase 7: Ética en IA
- ✅ Clase 8: Docker y MLOps

### Semanas 9-10: Proyecto Final
- 🎓 Prueba Técnica
- 📊 Presentación

## 💡 Consejos de Estudio

### Buenas Prácticas
1. **Práctica Diaria**: Dedica al menos 1-2 horas diarias
2. **Toma Notas**: Documenta lo que aprendes
3. **Experimenta**: Modifica el código y observa qué pasa
4. **Pregunta**: Usa los foros y comunidades
5. **Proyectos Personales**: Aplica lo aprendido en tus propios proyectos

### Gestión de Tiempo
- 📅 Establece un horario fijo de estudio
- ⏰ Usa técnica Pomodoro (25 min trabajo, 5 min descanso)
- 📝 Lleva un registro de tu progreso
- 🎯 Establece metas semanales

## 🆘 Solución de Problemas Comunes

### Error: "Module not found"
```bash
# Asegúrate de tener el entorno virtual activado
pip install nombre_del_paquete
```

### Error: "Jupyter command not found"
```bash
# Reinstala jupyter
pip install --upgrade jupyter jupyterlab
```

### Kernel no aparece en Jupyter
```bash
# Registra el kernel del entorno virtual
python -m ipykernel install --user --name=venv
```

### Problemas con TensorFlow
```bash
# Para CPU solamente
pip install tensorflow-cpu

# Para GPU (requiere CUDA)
pip install tensorflow
```

## 📞 Soporte

### Recursos de Ayuda
- 📧 Correo del instructor: [consultar calendario]
- 💬 Foro de discusión: [link]
- 👥 Grupo de estudio: [link]
- 📚 Documentación oficial: Ver referencias en cada clase

### Horarios de Consulta
- Consultar con el instructor del diplomado
- Sesiones de Q&A: [por definir]

## 🎯 Checklist de Inicio

Marca cada item cuando lo completes:

- [ ] Python 3.8+ instalado y funcionando
- [ ] Repositorio clonado localmente
- [ ] Entorno virtual creado y activado
- [ ] Dependencias instaladas correctamente
- [ ] Jupyter Lab funcionando
- [ ] Primer notebook ejecutado exitosamente
- [ ] README principal leído completamente
- [ ] Estructura del repositorio explorada
- [ ] Primer ejercicio iniciado
- [ ] Herramientas de desarrollo configuradas

## 📈 Próximos Pasos

Una vez completada la configuración:

1. 📖 Lee el material de la Clase 1
2. 💻 Completa el primer notebook
3. 🏋️ Resuelve el primer ejercicio
4. 🤝 Únete a la comunidad del diplomado
5. 🎯 Define tus objetivos personales de aprendizaje

## 🌟 Recursos Adicionales

### Para Reforzar Python
- [Python.org Tutorial](https://docs.python.org/3/tutorial/)
- [Real Python](https://realpython.com/)
- [Python for Everybody](https://www.py4e.com/)

### Para Matemáticas
- [Khan Academy - Linear Algebra](https://www.khanacademy.org/math/linear-algebra)
- [Khan Academy - Statistics](https://www.khanacademy.org/math/statistics-probability)

### Práctica Adicional
- [Kaggle Competitions](https://www.kaggle.com/competitions)
- [LeetCode](https://leetcode.com/)
- [HackerRank](https://www.hackerrank.com/)

---

## ✨ ¡Estás Listo!

Ya tienes todo configurado para comenzar tu viaje en Machine Learning. 

**Recuerda:** La clave del éxito es la práctica constante. No te desanimes si algo no funciona a la primera, ¡es parte del proceso de aprendizaje!

🚀 **¡Adelante y mucho éxito!**

---

**Última actualización:** Octubre 2024  
**Versión:** 1.0
