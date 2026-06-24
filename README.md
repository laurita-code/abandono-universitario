# Sistema de alerta temprana de abandono universitario

Trabajo Final de la asignatura **Inteligencia Artificial y Estadística**.

## Objetivo del proyecto

Construir una pequeña aplicación que estime si un estudiante universitario va a acabar
**abandonando** la carrera o **graduándose**, a partir de unos pocos datos sencillos:
el turno (diurno/nocturno), si tiene beca, si debe matrícula, su edad al matricularse,
la nota de admisión y el porcentaje de asignaturas que ha aprobado en su primer curso.

## Qué problema resuelve la app

El abandono universitario es un problema real que preocupa a las universidades. La
aplicación imita un **sistema de alerta temprana**: una vez que un estudiante ha cursado
su primer año, estima su riesgo de abandono para poder ofrecerle ayuda a tiempo
(tutorías, becas, apoyo).

> **Importante (cuándo se predice):** la predicción NO se hace en el momento de admitir
> al alumno, sino DESPUÉS de su primer curso. Se comprobó con los datos que solo con el
> perfil de entrada (turno, beca, edad, nota de admisión) el modelo apenas acierta más
> que adivinando; lo que de verdad anticipa el abandono es el rendimiento del primer año.
> Por eso la barra lateral describe a un alumno que **ya ha cursado un año**.

La aplicación permite:

- Ver y limpiar los datos reales de los estudiantes.
- Ver gráficos del abandono según la carrera y la situación de beca.
- Entrenar un modelo sencillo (Regresión Logística).
- Describir a un estudiante y obtener una predicción en tiempo real
  (abandona / se gradúa) junto con la probabilidad.

## Datos utilizados (dos fuentes)

El proyecto importa datos desde **dos archivos locales** (no necesita internet durante
la ejecución):

1. `data/estudiantes.csv` — datos reales de 4.424 estudiantes del dataset público
   **"Predict Students' Dropout and Academic Success"** del UCI Machine Learning
   Repository (id 697). Se descarga sin necesidad de registro. El CSV usa el punto y
   coma (`;`) como separador y las columnas están en inglés.
2. `data/carreras.json` — diccionario propio que traduce el código numérico de cada
   carrera (por ejemplo `9500`, `171`) a su nombre legible (Enfermería, Animación y
   Diseño Multimedia...), según la documentación oficial del propio dataset.

> Procedencia del CSV: se descargó una sola vez desde el zip público del UCI
> (`https://archive.ics.uci.edu/static/public/697/`) y se guardó en `data/` para que la
> app no dependa de internet durante la exposición.

## Adaptaciones al sistema español

El dataset es de una universidad **portuguesa**, así que se hicieron dos ajustes para
que se entienda en España (ambos documentados en el código, dentro del Requisito 3):

- **Nota de admisión:** venía en escala 0-200; se pasó a la escala española **0-14**
  (los 10 de la nota de acceso + hasta 4 puntos de la fase voluntaria de la EBAU).
- **Asignaturas:** el dataset cuenta "unidades curriculares" (módulos); se transformó en
  el **porcentaje de asignaturas aprobadas** sobre las matriculadas, que es universal.

## Cómo reproducir el proyecto

El proyecto usa [uv](https://docs.astral.sh/uv/) para gestionar el entorno.

1. **Crear el entorno virtual** (con Python 3.11):

   ```bash
   uv venv --python 3.11
   ```

2. **Instalar las dependencias**:

   ```bash
   uv sync
   ```

   > Si prefieres usar `pip` en lugar de `uv`, puedes hacer:
   > `pip install -r requirements.txt`

3. **Ejecutar la aplicación**:

   ```bash
   streamlit run main.py
   ```

   Se abrirá automáticamente en el navegador (normalmente en
   `http://localhost:8501`).

## Estructura de archivos

```
data/
  estudiantes.csv   <- datos reales de los estudiantes (fuente CSV, UCI)
  carreras.json     <- nombres de las carreras (fuente JSON)
main.py             <- aplicación completa (se lee de arriba a abajo)
README.md
requirements.txt
pyproject.toml
```

## Despliegue en internet (Streamlit Community Cloud)

La aplicación se puede publicar gratis con una URL pública:

1. Sube este proyecto a un repositorio **público** de GitHub.
2. Entra en [share.streamlit.io](https://share.streamlit.io) e inicia sesión con GitHub.
3. Pulsa **"New app"**, elige el repositorio y el archivo principal `main.py`.
4. Streamlit Cloud instalará solo las dependencias de `requirements.txt` y
   generará una URL pública funcional.

## Los 8 requisitos del trabajo

1. **Definición del problema** — explicado al inicio de la app y de este README.
2. **Importación de datos** — CSV con `pandas` y JSON con la librería `json`.
3. **Transformaciones** — rellenar nulos de la nota, reescalar la nota a 0-14, filtrar
   atípicos de edad, descartar a los aún matriculados y crear la columna `TasaAprobado`.
4. **Mapeo** — `.map()` para el turno (texto) y para el nombre de la carrera (JSON).
5. **Ordenación** — tabla ordenada con `sort_values()` y un selector de criterio.
6. **Visualización** — histograma de edades y gráficos de abandono por carrera y por beca.
7. **Modelización** — `train_test_split`, `LogisticRegression` y `accuracy_score`.
8. **Comunicación** — barra lateral interactiva con un botón "Predecir".
