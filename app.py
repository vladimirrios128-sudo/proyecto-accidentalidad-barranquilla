"""
App Streamlit: Accidentalidad en Barranquilla
Predice si un accidente de tránsito deja víctimas (heridos o muertos) o solo daños.
Autora: Miguel Angel Tunjuelo, ladimir Rios
Fundamentos de Ciencia de Datos - Universidad de San Buenaventura
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

st.set_page_config(page_title="Accidentalidad Barranquilla", page_icon="🚦", layout="wide")
sns.set_theme(style="whitegrid")

ARCHIVO = "accidentalidad_barranquilla.csv"
CAT_COLS = ["clase", "tipo_via", "franja", "dia"]
NUM_COLS = ["hora_dia", "mes_num", "fin_de_semana"]
MESES = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, "July": 7,
         "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
DIAS_ES = {"Mon": "Lunes", "Tue": "Martes", "Wed": "Miércoles", "Thu": "Jueves",
           "Fri": "Viernes", "Sat": "Sábado", "Sun": "Domingo"}
MESES_ES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto",
            "Septiembre", "Octubre", "Noviembre", "Diciembre"]


# ----------------------------------------------------------------------------
# Funciones de limpieza y preparación (las mismas del notebook)
# ----------------------------------------------------------------------------
def tipo_via(sitio):
    primera = str(sitio).upper().replace(".", " ").split()
    primera = primera[0] if primera else ""
    if primera in ["CL", "CALLE", "CLLE", "CLL", "CALE", "CALL"]:
        return "Calle"
    if primera in ["CR", "CARRERA", "CRA", "KR", "CARERA", "CARERRA", "KRA", "CRR"]:
        return "Carrera"
    if primera in ["AV", "AVENIDA", "AVE", "CIRCUNVALAR", "CORDIALIDAD", "TRONCAL"]:
        return "Avenida"
    if primera in ["VIA", "KM", "KILOMETRO"]:
        return "Vía/Carretera"
    if primera in ["DG", "DIAGONAL", "DIAG"]:
        return "Diagonal"
    if primera in ["TV", "TRANSVERSAL", "TRANSV"]:
        return "Transversal"
    return "Otro"


def franja(h):
    if 0 <= h < 6:
        return "Madrugada"
    if 6 <= h < 12:
        return "Mañana"
    if 12 <= h < 18:
        return "Tarde"
    return "Noche"


@st.cache_data
def cargar_datos():
    df = pd.read_csv(ARCHIVO)
    df.columns = ["fecha", "hora", "gravedad", "clase", "sitio",
                  "heridos", "muertos", "cantidad", "anio", "mes", "dia"]
    # Limpieza
    df["heridos"] = df["heridos"].fillna(0).astype(int)
    df["muertos"] = df["muertos"].fillna(0).astype(int)
    df = df.drop_duplicates(subset=["fecha", "hora", "gravedad", "clase", "sitio", "heridos", "muertos"])
    df["clase"] = df["clase"].replace({"Incendio": "Otro"})
    df["hora_dia"] = pd.to_datetime(df["hora"].str.strip(), format="%I:%M:%S:%p", errors="coerce").dt.hour
    df = df.dropna(subset=["hora_dia"])
    df["hora_dia"] = df["hora_dia"].astype(int)
    # Preparación
    df["tipo_via"] = df["sitio"].apply(tipo_via)
    df["franja"] = df["hora_dia"].apply(franja)
    df["fin_de_semana"] = df["dia"].isin(["Sat", "Sun"]).astype(int)
    df["mes_num"] = df["mes"].map(MESES)
    df["con_victimas"] = (df["gravedad"] != "Solo daños").astype(int)
    return df


@st.cache_resource
def entrenar_modelos(df_modelo):
    X = df_modelo[CAT_COLS + NUM_COLS]
    y = df_modelo["con_victimas"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("num", StandardScaler(), NUM_COLS),
    ])
    modelos = {
        "Regresión logística": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Árbol de decisión": DecisionTreeClassifier(max_depth=6, min_samples_leaf=50,
                                                    class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=10,
                                                class_weight="balanced", random_state=42, n_jobs=-1),
    }
    entrenados, filas, matrices = {}, [], {}
    for nombre, modelo in modelos.items():
        pipe = Pipeline([("prep", prep), ("modelo", modelo)]).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        entrenados[nombre] = pipe
        matrices[nombre] = confusion_matrix(y_test, pred)
        filas.append({"Modelo": nombre,
                      "Accuracy": accuracy_score(y_test, pred),
                      "Precision": precision_score(y_test, pred),
                      "Recall": recall_score(y_test, pred),
                      "F1": f1_score(y_test, pred)})
    tabla = pd.DataFrame(filas).set_index("Modelo").round(3)
    return entrenados, tabla, matrices


df = cargar_datos()
df_modelo = df[df["anio"] <= 2022]  # 2023+ casi no registra accidentes de solo daños
modelos, tabla, matrices = entrenar_modelos(df_modelo)
mejor = tabla["F1"].idxmax()

# ----------------------------------------------------------------------------
# Interfaz
# ----------------------------------------------------------------------------
st.sidebar.title("🚦 Accidentalidad BAQ")
pagina = st.sidebar.radio("Ir a:", ["Inicio", "Exploración de datos", "Modelos", "Predicción"])
st.sidebar.markdown("---")
st.sidebar.caption("Ladimir Rios Doza y Miguel Tunjuelo. \n\nFundamentos de Ciencia de Datos\n\nUniversidad de San Buenaventura")

if pagina == "Inicio":
    st.title("🚦 ¿El accidente deja víctimas?")
    st.subheader("Accidentalidad vial en Barranquilla (2018 – 2026)")
    st.markdown("""
Esta aplicación usa **machine learning** para estimar si un accidente de tránsito en Barranquilla
**deja víctimas (heridos o muertos)** o **solo daños materiales**, a partir de la clase de accidente,
la hora, el día, el mes y el tipo de vía.

**¿Por qué importa para Colombia?** Los siniestros viales son un problema de salud pública.
Estimar desde el primer reporte qué accidentes probablemente tienen víctimas ayuda a **priorizar
ambulancias** y a enfocar **controles y campañas** en los horarios y lugares más críticos.
""")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accidentes registrados", f"{len(df):,}".replace(",", "."))
    c2.metric("Con heridos", f"{(df['gravedad'] == 'Con heridos').sum():,}".replace(",", "."))
    c3.metric("Con muertos", f"{(df['gravedad'] == 'Con muertos').sum():,}".replace(",", "."))
    c4.metric("Modelo elegido", mejor)
    st.markdown("**Fuente:** Datos Abiertos Colombia – *Accidentalidad en Barranquilla*.")
    with st.expander("Ver una muestra de los datos"):
        st.dataframe(df[["fecha", "hora", "gravedad", "clase", "sitio", "heridos", "muertos"]].head(50))

elif pagina == "Exploración de datos":
    st.title("📊 Exploración de datos")
    anios = sorted(df["anio"].unique())
    rango = st.slider("Años", int(min(anios)), int(max(anios)), (int(min(anios)), int(max(anios))))
    d = df[df["anio"].between(*rango)]
    st.caption(f"{len(d):,} accidentes en el rango seleccionado".replace(",", "."))

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        d["gravedad"].value_counts().plot(kind="barh", color="steelblue", ax=ax)
        ax.set_title("Accidentes por gravedad"); ax.set_xlabel("Cantidad")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        d["clase"].value_counts().plot(kind="barh", color="darkorange", ax=ax)
        ax.set_title("Accidentes por clase"); ax.set_xlabel("Cantidad")
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(d["hora_dia"], bins=24, discrete=True, color="teal", ax=ax)
        ax.set_title("Accidentes por hora del día"); ax.set_xlabel("Hora (0-23)")
        st.pyplot(fig)
    with col4:
        fig, ax = plt.subplots(figsize=(6, 4))
        (d.groupby("hora_dia")["con_victimas"].mean() * 100).plot(marker="o", color="crimson", ax=ax)
        ax.set_title("% con víctimas según la hora"); ax.set_xlabel("Hora (0-23)"); ax.set_ylabel("%")
        st.pyplot(fig)

    col5, col6 = st.columns(2)
    with col5:
        fig, ax = plt.subplots(figsize=(6, 4))
        df.groupby("anio").size().plot(kind="bar", color="slategray", ax=ax)
        ax.set_title("Accidentes por año (todos los años)"); ax.set_xlabel("Año")
        st.pyplot(fig)
    with col6:
        fig, ax = plt.subplots(figsize=(6, 4))
        (df.groupby("anio")["con_victimas"].mean() * 100).plot(kind="bar", color="purple", ax=ax)
        ax.set_title("% con víctimas por año (todos los años)"); ax.set_xlabel("Año"); ax.set_ylabel("%")
        st.pyplot(fig)
    st.info("Desde 2023 casi todos los registros son con víctimas: los accidentes de solo daños "
            "prácticamente dejaron de registrarse. Por eso el modelo se entrena con 2018–2022.")

    fig, ax = plt.subplots(figsize=(10, 3))
    sns.boxplot(x=d.loc[d["heridos"] > 0, "heridos"], color="lightblue", ax=ax)
    ax.set_title("Box plot: heridos por accidente (outliers)")
    st.pyplot(fig)

elif pagina == "Modelos":
    st.title("🤖 Comparación de modelos")
    st.markdown("Se entrenaron 3 modelos con los accidentes de **2018–2022** (80% entrenamiento, 20% prueba).")
    st.dataframe(tabla.style.highlight_max(axis=0, color="#c6efce"))
    fig, ax = plt.subplots(figsize=(9, 4))
    tabla.plot(kind="bar", ax=ax, colormap="viridis")
    ax.set_ylim(0, 1); ax.set_title("Métricas por modelo"); plt.xticks(rotation=0)
    st.pyplot(fig)

    st.subheader("Matrices de confusión")
    cols = st.columns(3)
    for col, (nombre, cm) in zip(cols, matrices.items()):
        with col:
            fig, ax = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                        xticklabels=["Solo daños", "Víctimas"], yticklabels=["Solo daños", "Víctimas"])
            ax.set_title(nombre); ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
            st.pyplot(fig)
    st.success(f"**Modelo elegido: {mejor}.** Tiene el mejor F1 y el mejor Recall: detecta más "
               "accidentes que realmente tuvieron víctimas, que es el error más costoso de cometer.")

else:
    st.title("🔮 Predicción")
    st.markdown("Ingresa los datos de un accidente para estimar la probabilidad de que tenga víctimas.")
    c1, c2 = st.columns(2)
    with c1:
        clase = st.selectbox("Clase de accidente", sorted(df_modelo["clase"].unique()),
                             index=sorted(df_modelo["clase"].unique()).index("Choque"))
        via = st.selectbox("Tipo de vía", ["Calle", "Carrera", "Avenida", "Vía/Carretera",
                                           "Diagonal", "Transversal", "Otro"])
        hora = st.slider("Hora del día (0-23)", 0, 23, 18)
    with c2:
        dia_es = st.selectbox("Día de la semana", list(DIAS_ES.values()))
        mes_es = st.selectbox("Mes", MESES_ES)
        modelo_sel = st.selectbox("Modelo", list(modelos.keys()), index=list(modelos.keys()).index(mejor))

    dia = [k for k, v in DIAS_ES.items() if v == dia_es][0]
    entrada = pd.DataFrame([{
        "clase": clase, "tipo_via": via, "franja": franja(hora), "dia": dia,
        "hora_dia": hora, "mes_num": MESES_ES.index(mes_es) + 1,
        "fin_de_semana": int(dia in ["Sat", "Sun"]),
    }])

    if st.button("Predecir", type="primary"):
        prob = modelos[modelo_sel].predict_proba(entrada)[0][1]
        pred = modelos[modelo_sel].predict(entrada)[0]
        if pred == 1:
            st.error(f"⚠️ Probablemente **CON VÍCTIMAS** (puntaje del modelo: {prob:.0%})")
        else:
            st.success(f"✅ Probablemente **SOLO DAÑOS** (puntaje de víctimas: {prob:.0%})")
        st.progress(float(prob))
        st.caption("El puntaje viene de un modelo entrenado con class_weight='balanced', por eso se usa "
                   "para comparar casos más que como una probabilidad exacta. Es una herramienta de apoyo "
                   "para priorizar la atención, no reemplaza la valoración en el sitio.")
