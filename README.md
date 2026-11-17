# Accidentalidad en Barranquilla: ¿el accidente deja víctimas?

Proyecto final de **Fundamentos de Ciencia de Datos** – Universidad de San Buenaventura.

## Integrantes
- miguel angel tunjuelo
- ladimir rios

## Descripción
Aplicación y notebook de ciencia de datos que, a partir de los registros de accidentes de tránsito de Barranquilla (2018–2026), entrena modelos de **machine learning** para predecir si un accidente **deja víctimas (heridos o muertos)** o **solo daños materiales**.
Este proyecto sigue el pipeline completo trabajado en el semestre: definición del problema → exploración → limpieza → preparación → modelado (3 modelos) → evaluación → selección del modelo → conclusiones.

## Problemática
Los siniestros viales son una de las principales causas de muerte violenta en Colombia y un problema de salud pública. En Barranquilla se registraron cerca de 29.000 accidentes entre 2018 y 2026

**Pregunta:** con la información disponible en el momento del reporte (clase de accidente, hora, día, mes y tipo de vía), podemos estimar si el accidente tiene víctimas?

**Utilidad:**
- Priorizar el envío de **ambulancias** y agentes de tránsito.
- Identificar **horarios, días y tipos de accidente** más peligrosos para orientar controles y campañas de la Secretaría de Tránsito.

## Contenido del repositorio
| Archivo | Descripción |
|---|---|
| `README.md` | Este documento |
| `notebook.ipynb` | Notebook documentado y comentado celda por celda con todo el pipeline |
| `app.py` | Código completo de la aplicación en Streamlit |
| `requirements.txt` | Librerías necesarias para correr la app |
| `accidentalidad_barranquilla.csv` | Dataset (fuente: [Datos Abiertos Colombia – Accidentalidad en Barranquilla](https://www.datos.gov.co/)) |

## Dataset
- **Registros:** 28.865 accidentes (2018 – 2026)
- **Columnas:** fecha, hora, gravedad, clase de accidente, sitio, cantidad de heridos, cantidad de muertos, cantidad de accidentes, año, mes y día.
- **Variable objetivo:** `con_victimas` = 1 si la gravedad es *Con heridos* o *Con muertos*; 0 si es *Solo daños*.

## Metodología (resumen)
1. **Exploración:** estadísticas descriptivas, distribuciones, box plots, detección de nulos y outliers, regresión lineal de la tendencia mensual.
2. **Limpieza:** nulos de heridos/muertos imputados con 0 (el nulo significa "ninguno"); eliminación de duplicados; normalización de direcciones a tipo de vía; se usan solo los años **2018–2022** porque desde 2023 casi no se registran accidentes de solo daños.
3. **Preparación:** variables nuevas (franja horaria, fin de semana, tipo de vía), One-Hot Encoding, StandardScaler, división 80/20 estratificada.
4. **Modelado:** Regresión logística, Árbol de decisión y Random Forest (con `class_weight="balanced"`).
5. **Evaluación:** Accuracy, Precision, Recall, F1 y matriz de confusión.

| Modelo | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Regresión logística | 0.694 | 0.569 | 0.353 | 0.436 |
| **Árbol de decisión** | 0.654 | 0.485 | **0.532** | **0.507** |
| Random Forest | 0.681 | 0.527 | 0.436 | 0.477 |

**Modelo elegido: Árbol de decisión** – mejor F1 y mejor Recall (detecta más accidentes con víctimas), además de ser fácil de interpretar.

## Conclusion
- Los **atropellos**, **caídas de ocupante** y **volcamientos** casi siempre dejan víctimas; los choques son mayoritariamente solo daños.
- En la **madrugada** el ~67% de los accidentes tiene víctimas, frente a ~40% en la tarde.
- El **domingo** hay menos accidentes, pero una mayor proporción con víctimas (~58%).
- Recomendación: reforzar controles de velocidad y alcoholemia en noches, madrugadas y fines de semana, y priorizar la seguridad de peatones y motociclistas.

## ▶️ Instrucciones de uso

## Link de la aplicación
https://proyecto-accidentalidad-barranquilla-on4hwgfhmf2y4v5jdrv3mn.streamlit.app
### Usar la app en línea
Entrar al link de la app (arriba), elegir una sección en el menú lateral:
- **Inicio:** descripción del problema y cifras generales.
- **Exploración de datos:** gráficas interactivas filtradas por año.
- **Modelos:** comparación de los 3 modelos y matrices de confusión.
- **Predicción:** ingresar clase de accidente, tipo de vía, hora, día y mes, y presionar **Predecir**.
