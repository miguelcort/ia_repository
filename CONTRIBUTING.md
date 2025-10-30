# Guía de Contribución

¡Gracias por tu interés en contribuir al repositorio del Diplomado en Machine Learning!

## Cómo Contribuir

### Reportar Errores
Si encuentras un error:
1. Verifica que no esté ya reportado en los Issues
2. Crea un nuevo Issue con una descripción clara
3. Incluye pasos para reproducir el error
4. Añade capturas de pantalla si es relevante

### Sugerir Mejoras
Para sugerir mejoras o nuevas funcionalidades:
1. Abre un Issue describiendo tu propuesta
2. Explica por qué sería útil
3. Proporciona ejemplos si es posible

### Contribuir con Código

#### 1. Fork y Clone
```bash
# Fork el repositorio en GitHub
# Luego clona tu fork
git clone https://github.com/TU-USUARIO/ia_repository.git
cd ia_repository
```

#### 2. Crear una Rama
```bash
git checkout -b feature/mi-contribucion
```

#### 3. Hacer Cambios
- Sigue el estilo de código existente
- Comenta tu código cuando sea necesario
- Actualiza la documentación relevante

#### 4. Commit
```bash
git add .
git commit -m "Descripción clara de los cambios"
```

#### 5. Push y Pull Request
```bash
git push origin feature/mi-contribucion
```
Luego crea un Pull Request en GitHub.

## Estándares de Código

### Python
- Seguir PEP 8
- Usar nombres descriptivos de variables
- Documentar funciones con docstrings
- Máximo 100 caracteres por línea

### Notebooks
- Título descriptivo en la primera celda
- Markdown para explicaciones
- Limpiar outputs antes de commit
- Orden lógico de celdas

### Documentación
- README en cada carpeta principal
- Español como idioma principal
- Ejemplos claros y concisos
- Enlaces a recursos externos cuando sea apropiado

## Estructura de Commits

Formato recomendado:
```
tipo: descripción breve

Descripción más detallada si es necesario.
```

Tipos:
- `feat`: Nueva funcionalidad
- `fix`: Corrección de errores
- `docs`: Cambios en documentación
- `style`: Formato, espacios, etc.
- `refactor`: Refactorización de código
- `test`: Añadir o modificar tests
- `chore`: Mantenimiento general

Ejemplos:
```
feat: agregar notebook de regresión logística

docs: actualizar README de Clase 3

fix: corregir error en carga de datos
```

## Revisión de Código

Todos los Pull Requests serán revisados:
- Funcionalidad correcta
- Código limpio y legible
- Documentación adecuada
- Sin conflictos de merge

## Licencia

Al contribuir, aceptas que tu código se licencie bajo los mismos términos que el proyecto.

## Preguntas

Si tienes preguntas, abre un Issue o contacta a los mantenedores del repositorio.

## Código de Conducta

### Nuestro Compromiso
Este es un espacio de aprendizaje inclusivo y respetuoso.

### Comportamiento Esperado
- Usar lenguaje acogedor e inclusivo
- Respetar diferentes puntos de vista
- Aceptar críticas constructivas
- Enfocarse en lo mejor para la comunidad

### Comportamiento Inaceptable
- Lenguaje o imágenes inapropiadas
- Acoso de cualquier tipo
- Publicar información privada de otros
- Conducta poco profesional

## Reconocimientos

Los contribuidores serán reconocidos en el README principal.

---

¡Gracias por ayudar a mejorar este proyecto educativo! 🚀
