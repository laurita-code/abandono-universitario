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
st.header("DEFINICIÓN DEL PROBLEMA")

st.markdown("""
**¿Qué queremos predecir? :** El proyecto consiste en el desarrollo de un Sistema de alerta 
temprana para la prevención del abandono universitario. El objetivo principal es predecir de forma automatizada y 
en base a los datos obtenidos, si un estudiante que ya ha cursado su primer año va a acabar
abandonando la carrera o, por el contrario, va a graduarse.

**¿Cuándo se predice? :** NO en el momento de admitir al alumno, sino DESPUÉS
de su primer curso. ¿Por qué en ese momento y no antes? Comprobamos con los datos que solo
con el perfil de entrada (turno, beca, edad, nota de admisión) el modelo apenas acierta
más que adivinando; en cambio, en cuanto se sabe CÓMO le ha ido el primer año (cuántas
asignaturas aprueba), el acierto sube muchísimo. Es decir, lo que de verdad anticipa el
abandono no es de dónde viene el alumno, sino su rendimiento en el primer curso.

**¿Por qué es útil? :** Es justo lo que hacen las universidades con los "sistemas de alerta
temprana". Al identificar a los estudiantes en situación de riesgo inmediatamente después de 
concluir su primer año, la universidad dispone de tiempo suficiente para actuar. 
Esto permite diseñar y aplicar estrategias de apoyo como tutorías de refuerzo, asignación de becas 
específicas o programas de acompañamiento psicológico y académico antes de que el estudiante tome ç
la decisión de abandonar la carrera. 

**Objetivo concreto:** Estimar el riesgo de abandono a partir del turno (día/noche),
si el estudiante tiene beca, si debe matrícula, su edad al matricularse, la nota de
admisión y el porcentaje de asignaturas que ha aprobado en su primer curso.
""")

# REQUISITO 2: IMPORTACIÓN DE DATOS (dos fuentes)
st.header("REQUISITO 2: IMPORTACIÓN DE DATOS")
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
st.header("REQUISITO 3: TRANSFORMACIONES")

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

# REQUISITO 4: MAPEO con .map()
st.header("REQUISITO 4: MAPEO")

datos["Turno"] = datos["Daytime/evening attendance"].map({1: "Diurno", 0: "Nocturno"})
st.write(datos.head())

datos["Carrera"] = datos["Course"].astype(str).map(carreras)

st.write("Ejemplo del mapeo realizado (primeras filas):")
st.dataframe(datos[["Daytime/evening attendance", "Turno", "Course", "Carrera"]].head())

# REQUISITO 5: ORDENACIÓN
st.header("REQUISITO 5: ORDENACIÓN")

criterio = st.selectbox(
    "Ordenar los estudiantes por:",
    ["Nota de admisión (de mayor a menor)", "Edad (de mayor a menor)","Tasa de Aprobado primer año (de mayor a menor)"]
)

if criterio == "Nota de admisión (de mayor a menor)":
    datos_ordenados = datos.sort_values(by="NotaAdmision", ascending=False)

elif criterio == "Edad (de mayor a menor)":
    datos_ordenados = datos.sort_values(by="Age at enrollment", ascending=False)

else: 
    datos_ordenados = datos.sort_values(by="TasaAprobado", ascending=False)

st.write("Los 10 primeros estudiantes según el criterio elegido:")
st.dataframe(datos_ordenados[["Carrera", "Turno", "Age at enrollment", "NotaAdmision", "TasaAprobado"]].head(10))

# REQUISITO 6: VISUALIZACIÓN


figura1, ejes1 = plt.subplots()
sns.histplot(datos["Age at enrollment"], bins=20, color="skyblue", ax=ejes1)
ejes1.set_title("Distribución de la edad al matricularse")
ejes1.set_xlabel("Edad")
ejes1.set_ylabel("Número de estudiantes")
st.pyplot(figura1)

abandono_por_carrera = datos.groupby("Carrera")["Abandona"].mean()

figura2, ejes2 = plt.subplots()
ejes2.bar(abandono_por_carrera.index, abandono_por_carrera.values, color="orange")
ejes2.set_title("Tasa de abandono por carrera")
ejes2.set_xlabel("Carrera")
ejes2.set_ylabel("Proporción que abandona")
plt.xticks(rotation=45, ha="right")
st.pyplot(figura2)

abandono_por_beca = datos.groupby("Scholarship holder")["Abandona"].mean()

figura3, ejes3 = plt.subplots()
ejes3.bar(["Sin beca", "Con beca"], abandono_por_beca.values, color="green")
ejes3.set_title("Tasa de abandono según beca")
ejes3.set_xlabel("Situación de beca")
ejes3.set_ylabel("Proporción que abandona")
st.pyplot(figura3)

columnas_candidatas = ["Daytime/evening attendance", "Scholarship holder", "Debtor",
                       "Age at enrollment", "NotaAdmision", "TasaAprobado"]
correlacion_abandono = datos[columnas_candidatas + ["Abandona"]].corr()["Abandona"].drop("Abandona")

figura4, ejes4 = plt.subplots()
ejes4.barh(correlacion_abandono.index, correlacion_abandono.values, color="purple")
ejes4.set_title("Relación de cada variable con el abandono (correlación)")
ejes4.set_xlabel("Coeficiente de correlación con 'Abandona'")
st.pyplot(figura4)

# REQUISITO 7: MODELIZACIÓN
st.header("REQUISITO 7: MODELIZACIÓN")

columnas_entrada = ["Daytime/evening attendance", "Scholarship holder", "Debtor",
                    "Age at enrollment", "NotaAdmision", "TasaAprobado"]
X = datos[columnas_entrada]
y = datos["Abandona"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)
acierto = accuracy_score(y_test, predicciones)

st.write("El modelo ya está entrenado.")
st.write("Acierto del modelo sobre los datos de prueba:", round(acierto * 100, 1), "%")

# REQUISITO 8: COMUNICACIÓN (interactividad con el usuario)
st.sidebar.title("Describe a un estudiante")
st.sidebar.caption("Describe a un alumno que YA ha cursado su primer año.")

turno_elegido = st.sidebar.selectbox("Turno", ["Diurno", "Nocturno"])
beca_elegida = st.sidebar.selectbox("Tiene beca", ["No", "Sí"])
deudor_elegido = st.sidebar.selectbox("Debe matrícula", ["No", "Sí"])
edad_elegida = st.sidebar.slider("Edad al matricularse", 17, 60, 20)
nota_elegida = st.sidebar.slider("Nota de admisión (sobre 14)", 0.0, 14.0, 9.0)
tasa_elegida = st.sidebar.slider("Porcentaje de asignaturas aprobadas", 0, 100, 80)

if turno_elegido == "Diurno":
    turno_numero = 1
else:
    turno_numero = 0

if beca_elegida == "Sí":
    beca_numero = 1
else:
    beca_numero = 0

if deudor_elegido == "Sí":
    deudor_numero = 1
else:
    deudor_numero = 0


    
if st.sidebar.button("Predecir"):

    estudiante_nuevo = pd.DataFrame([[turno_numero, beca_numero, deudor_numero, edad_elegida, nota_elegida, tasa_elegida]],
                                    columns=columnas_entrada)

    resultado = modelo.predict(estudiante_nuevo)

    probabilidad = modelo.predict_proba(estudiante_nuevo)
    probabilidad_abandonar = probabilidad[0][1]

    if resultado[0] == 1:
        st.error("Este estudiante PROBABLEMENTE ABANDONARÍA los estudios.")
    else:
        st.success("Este estudiante PROBABLEMENTE SE GRADUARÍA.")

    st.write("Probabilidad de abandonar:", round(probabilidad_abandonar * 100, 1), "%")