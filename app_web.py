import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import requests
import joblib
import matplotlib.pyplot as plt
import shap
import streamlit.components.v1 as components

from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


# =========================
# Config
# =========================
API_URL = os.environ.get("API_URL", "http://localhost:5000/predict")

FEATURE_NAMES = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
FEATURE_LABELS = ["Longitud del sépalo (cm)", "Ancho del sépalo (cm)", "Longitud del pétalo (cm)", "Ancho del pétalo (cm)"]
CLASS_NAMES = ["Setosa", "Versicolor", "Virginica"]  # 0,1,2


# =========================
# Page
# =========================
st.set_page_config(page_title="Iris – Predicción + XAI", layout="wide")
st.title("API de Predicción del Modelo Iris + Explainability (SHAP + LIME)")
st.write("Introduce las características de la flor para obtener una predicción vía **API Flask** y ver explicaciones **SHAP + LIME** en la web.")


# =========================
# Load local model/data for XAI (in web container)
# =========================
@st.cache_resource
def load_model_local():
    return joblib.load("model.pkl")

@st.cache_data
def load_iris_data():
    # Debe existir dentro del contenedor web
    df = pd.read_csv("data/iris_dataset.csv")
    X = df[FEATURE_NAMES].copy()
    y = df["target"].copy()
    return X, y

model_local = load_model_local()
X_bg, y_bg = load_iris_data()

# SHAP explainer: rápido para RandomForest / árboles
@st.cache_resource
def make_shap_explainer(_model):
    return shap.TreeExplainer(_model)

shap_explainer = make_shap_explainer(model_local)

# LIME explainer
@st.cache_resource
def make_lime_explainer(X_train):
    return LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=FEATURE_NAMES,
        class_names=CLASS_NAMES,
        mode="classification",
        discretize_continuous=True,
    )

lime_explainer = make_lime_explainer(X_bg)


# =========================
# GLOBAL explanations (cached)
# =========================
@st.cache_data
def compute_global_explanations(_X: pd.DataFrame, _y: pd.Series, seed: int = 42, n_shap: int = 120, n_repeats: int = 20):
    """
    Calcula (una sola vez y cachea):
      - 3 SHAP summary plots (uno por clase) a partir de un muestreo
      - Permutation importance global del modelo
    """
    # 1) Muestreo para SHAP summary (para que sea rápido)
    n = min(n_shap, len(_X))
    X_sample = _X.sample(n=n, random_state=seed)

    sv = shap_explainer.shap_values(X_sample)

    # Normalizamos formato de salida
    if isinstance(sv, list):
        shap_values_per_class = sv  # [class0, class1, class2]
    else:
        # array (n_samples, n_features, n_classes)
        shap_values_per_class = [sv[:, :, k] for k in range(sv.shape[2])]

    # 2) Permutation importance global (sobre un split fijo reproducible)
    X_train, X_test, y_train, y_test = train_test_split(
        _X, _y, test_size=0.3, random_state=seed, stratify=_y
    )
    perm = permutation_importance(
        model_local,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="accuracy",
        n_jobs=-1
    )

    return X_sample, shap_values_per_class, perm.importances_mean, perm.importances_std


# =========================
# UI – Inputs + Prediction
# =========================
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1) Input")
    sepal_length = st.slider(FEATURE_LABELS[0], 0.0, 10.0, 5.0, step=0.1)
    sepal_width  = st.slider(FEATURE_LABELS[1], 0.0, 10.0, 3.0, step=0.1)
    petal_length = st.slider(FEATURE_LABELS[2], 0.0, 10.0, 4.0, step=0.1)
    petal_width  = st.slider(FEATURE_LABELS[3], 0.0, 10.0, 1.0, step=0.1)

    x = np.array([sepal_length, sepal_width, petal_length, petal_width], dtype=float)
    x_df = pd.DataFrame([x], columns=FEATURE_NAMES)

    with st.expander("Ver input"):
        st.dataframe(x_df)

with col_right:
    st.subheader("2) Predicción (vía API)")
    st.caption(f"API_URL: `{API_URL}`")

    if st.button("Obtener Predicción", type="primary"):
        payload = {"features": x.tolist()}

        try:
            response = requests.post(
                API_URL,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=10
            )

            if response.status_code != 200:
                st.error(f"Error en la petición: {response.status_code} - {response.text}")
                st.stop()

            res = response.json()
            pred = int(res.get("prediction"))
            proba = res.get("probability", {})
            confidence = res.get("confidence", None)

            species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
            predicted_species = species_map.get(pred, "Desconocida")

            st.success(f"La predicción es: **{predicted_species}** (clase={pred})")
            if proba:
                st.write("Probabilidades:")
                st.json(proba)
            if confidence is not None:
                st.write(f"Confianza: **{float(confidence):.3f}**")

            # Guardamos clase objetivo para SHAP/LIME local
            st.session_state["pred_class"] = pred

        except requests.exceptions.RequestException as e:
            st.error(f"No se pudo conectar con la API. Asegúrate de que está en ejecución. Error: {e}")
            st.stop()


st.divider()
st.subheader("3) Explainability (SHAP + LIME)")

pred_class = st.session_state.get("pred_class", 0)

tabs = st.tabs(["Local (esta flor)", "Global (dataset)"])


# =========================
# LOCAL explanations
# =========================
with tabs[0]:
    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.markdown("### SHAP local")
        with st.spinner("Calculando SHAP local..."):
            sv = shap_explainer.shap_values(x_df)

        # Normalizamos a vector de la clase objetivo
        if isinstance(sv, list):
            shap_vec = np.array(sv[pred_class]).reshape(-1)
        else:
            shap_vec = np.array(sv[0, :, pred_class]).reshape(-1)

        fig, ax = plt.subplots()
        order = np.argsort(np.abs(shap_vec))[::-1]
        ax.barh([FEATURE_LABELS[i] for i in order], shap_vec[order])
        ax.invert_yaxis()
        ax.set_title(f"Contribuciones SHAP hacia: {CLASS_NAMES[pred_class]}")
        ax.set_xlabel("Impacto en el score del modelo")
        st.pyplot(fig, clear_figure=True)

        st.caption("Valores positivos empujan hacia la clase objetivo; negativos empujan en contra (hacia las otras clases).")

    with c2:
        st.markdown("### LIME local")
        with st.spinner("Calculando LIME..."):
            exp = lime_explainer.explain_instance(
                data_row=x,
                predict_fn=lambda z: model_local.predict_proba(pd.DataFrame(z, columns=FEATURE_NAMES)),
                num_features=4,
                top_labels=3,
            )

        components.html(exp.as_html(), height=520, scrolling=True)
        st.caption("LIME ajusta un modelo interpretable local (aprox. lineal) alrededor de este punto.")


# =========================
# GLOBAL explanations (cached, not recalculated on each prediction)
# =========================
with tabs[1]:
    st.markdown("### Global explanations (cacheado)")
    st.write("Primero mostramos **3 SHAP summary plots** (uno por clase) y luego el ranking único de **Permutation Importance**.")

    n_shap = st.slider("Muestras para SHAP summary", min_value=60, max_value=150, value=120, step=30)
    n_repeats = st.slider("Permutation repeats (más = más estable pero más lento)", 5, 30, 20, step=5)

    with st.spinner("Calculando explicaciones globales (solo la primera vez; luego cacheado)..."):
        X_sample, shap_values_per_class, perm_mean, perm_std = compute_global_explanations(
            X_bg, y_bg, n_shap=n_shap, n_repeats=n_repeats
        )

    # 1) SHAP summary por clase (3 plots)
    for k, name in enumerate(CLASS_NAMES):
        st.markdown(f"#### SHAP global (summary) – clase: **{name}**")
        fig = plt.figure()
        shap.summary_plot(
            shap_values_per_class[k],
            X_sample,
            feature_names=FEATURE_NAMES,
            show=False
        )
        st.pyplot(fig, clear_figure=True)

    st.divider()

    # 2) Permutation importance (un ranking global único)
    st.markdown("#### Permutation Feature Importance (global del modelo)")
    st.caption("Mide la caída de accuracy al permutar cada feature (ranking global único).")

    order = np.argsort(perm_mean)[::-1]
    fig2, ax = plt.subplots()
    ax.barh([FEATURE_LABELS[i] for i in order], perm_mean[order], xerr=perm_std[order])
    ax.invert_yaxis()
    ax.set_xlabel("Decrease in accuracy (mean over permutations)")
    ax.set_title("Permutation Importance (test split)")
    st.pyplot(fig2, clear_figure=True)
