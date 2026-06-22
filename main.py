# --- Librerías que vamos a usar ---
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# REQUISITO 1: DEFINICIÓN DEL PROBLEMA
st.title("Sistema de alerta temprana de abandono universitario")

st.markdown("""
**¿Qué queremos predecir?** Si un estudiante que YA ha cursado su primer año va a acabar
abandonando la carrera o, por el contrario, va a graduarse.

**¿Cuándo se predice? (importante):** NO en el momento de admitir al alumno, sino DESPUÉS
de su primer curso. ¿Por qué en ese momento y no antes? Comprobamos con los datos que solo
con el perfil de entrada (turno, beca, edad, nota de admisión) el modelo apenas acierta
más que adivinando; en cambio, en cuanto se sabe CÓMO le ha ido el primer año (cuántas
asignaturas aprueba), el acierto sube muchísimo. Es decir, lo que de verdad anticipa el
abandono no es de dónde viene el alumno, sino su rendimiento en el primer curso.

**¿Por qué es útil?** Es justo lo que hacen las universidades con los "sistemas de alerta
temprana": tras el primer curso, detectar a quién está en riesgo para ofrecerle ayuda
(tutorías, becas, apoyo) antes de que sea tarde.

**Objetivo concreto:** estimar el riesgo de abandono a partir del turno (diurno/nocturno),
si el estudiante tiene beca, si debe matrícula, su edad al matricularse, la nota de
admisión y el porcentaje de asignaturas que ha aprobado en su primer curso.
""")
# REQUISITO 2: IMPORTACIÓN DE DATOS (dos fuentes)
datos = pd.read_csv("data/estudiantes.csv", sep=";")

datos = datos.rename(columns={"Daytime/evening attendance\t": "Daytime/evening attendance"})

archivo_carreras = open("data/carreras.json", encoding="utf-8")
carreras = json.load(archivo_carreras)
archivo_carreras.close()

numero_filas = datos.shape[0]
numero_columnas = datos.shape[1]

st.header("1. Importación de datos")
st.write("El archivo CSV se ha cargado correctamente.")
st.write("Número de filas (estudiantes):", numero_filas)
st.write("Número de columnas:", numero_columnas)
st.write("Diccionario de carreras cargado del JSON:", carreras)

# REQUISITO 3: TRANSFORMACIONES
st.header("2. Transformaciones de los datos")

# 1) Rellenar posibles notas vacías con la media
nota_media = datos["Admission grade"].mean()
datos["Admission grade"] = datos["Admission grade"].fillna(nota_media)

# 2) Reescalar la nota de 0-200 (escala portuguesa) a 0-14 (escala española)
datos["NotaAdmision"] = datos["Admission grade"] / 200 * 14

# 3) Filtrar casos atípicos de edad
datos = datos[datos["Age at enrollment"] <= 60]
st.write("Se han quitado los casos atípicos de edad (estudiantes de más de 60 años).")
st.write("Filas restantes después de filtrar los menores de 60 años:", datos.shape[0])

# 4) Quedarnos solo con casos cerrados y crear la columna objetivo
datos = datos[datos["Target"].isin(["Dropout", "Graduate"])]
datos["Abandona"] = datos["Target"].map({"Graduate": 0, "Dropout": 1})
st.write("Nos quedamos solo con estudiantes que abandonaron o se graduaron (descartamos los aún matriculados).")
st.write("Filas restantes después de filtrar los graduados y abandonos:", datos.shape[0])

# 5) Variable derivada: porcentaje de asignaturas aprobadas
aprobadas_total = datos["Curricular units 1st sem (approved)"] + datos["Curricular units 2nd sem (approved)"]
matriculadas_total = datos["Curricular units 1st sem (enrolled)"] + datos["Curricular units 2nd sem (enrolled)"]

datos["TasaAprobado"] = aprobadas_total / matriculadas_total * 100
datos["TasaAprobado"] = datos["TasaAprobado"].fillna(0)

st.write("La nota de admisión se ha pasado de la escala portuguesa (0-200) a la española (0-14).")


st.write("Se ha creado 'TasaAprobado' (porcentaje de asignaturas aprobadas sobre las matriculadas).")
st.write(datos)
