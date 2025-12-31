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
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


API_URL = os.environ.get("API_URL", "http://localhost:5000/predict")

FEATURE_NAMES = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
FEATURE_LABELS = ["Longitud del sépalo (cm)", "Ancho del sépalo (cm)", "Longitud del pétalo (cm)", "Ancho del pétalo (cm)"]
CLASS_NAMES = ["Setosa", "Versicolor", "Virginica"]  # 0,1,2

st.set_page_config(page_title="Iris – Predicción + XAI", layout="wide")
st.title("API de Predicción del Modelo Iris + Explainability")
st.write("Introduce las características de la flor para obtener una predicción")

# Carga del modelo y de datos
@st.cache_resource
def load_model_local():
    return joblib.load("model.pkl")

@st.cache_data
def load_iris_data():
    # Tiene que estar dentro del contenedor
    df = pd.read_csv("data/iris_dataset.csv")
    X = df[FEATURE_NAMES].copy()
    y = df["target"].copy()
    return X, y

model_local = load_model_local()
X_bg, y_bg = load_iris_data()

# SHAP explainer
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

@st.cache_data
def compute_local_explanations(x_tuple, pred_class: int):
    """
    Calcula y cachea:
      - shap_vec para la clase pred_class
      - HTML de LIME
    """
    x = np.array(x_tuple, dtype=float)
    x_df = pd.DataFrame([x], columns=FEATURE_NAMES)

    # SHAP local
    sv = shap_explainer.shap_values(x_df)
    if isinstance(sv, list):
        shap_vec = np.array(sv[pred_class]).reshape(-1)
    else:
        shap_vec = np.array(sv[0, :, pred_class]).reshape(-1)

    # LIME local
    exp = lime_explainer.explain_instance(
        data_row=x,
        predict_fn=lambda z: model_local.predict_proba(pd.DataFrame(z, columns=FEATURE_NAMES)),
        num_features=4,
        top_labels=3,
    )
    lime_html = exp.as_html()

    return shap_vec, lime_html

@st.cache_data
def feature_ablation_cv(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = 42,
    n_estimators: int = 200,
    n_splits: int = 5,
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    feature_sets = []
    
    # Baseline
    feature_sets.append((" (none)", list(X.columns)))

    # Quitando 1 feature cada vez
    for f in X.columns:
        cols = [c for c in X.columns if c != f]
        feature_sets.append((f, cols))

    # Quitando las dos del pétalo
    if "petal length (cm)" in X.columns and "petal width (cm)" in X.columns:
        cols = [c for c in X.columns if c not in ["petal length (cm)", "petal width (cm)"]]
        feature_sets.append(("petal length + petal width", cols))

    rows = []

    for removed_name, cols in feature_sets:
        scores = []
        for train_idx, test_idx in skf.split(X, y):
            Xtr, Xte = X.iloc[train_idx][cols], X.iloc[test_idx][cols]
            ytr, yte = y.iloc[train_idx], y.iloc[test_idx]

            m = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
            m.fit(Xtr, ytr)
            pred = m.predict(Xte)
            scores.append(accuracy_score(yte, pred))

        rows.append({
            "features_removed": removed_name,
            "cv_accuracy_mean": float(np.mean(scores)),
            "cv_accuracy_std": float(np.std(scores)),
        })

    df = pd.DataFrame(rows)

    base = df.loc[df["features_removed"].str.strip() == "(none)", "cv_accuracy_mean"].values[0]
    df["delta_vs_base"] = df["cv_accuracy_mean"] - base

    # Ordena por caida de la accuracy
    df = df.sort_values("delta_vs_base")
    return df

@st.cache_data
def sanity_check_local_perturbation_shap(
    x_tuple,
    _model,                
    _shap_explainer,         
    X_ref: pd.DataFrame,
    pred_class: int,
    top_k: int = 2,
    mode: str = "median", 
    q_low: float = 0.10, 
    q_high: float = 0.90
):
    
    x = np.array(x_tuple, dtype=float)
    x_df = pd.DataFrame([x], columns=FEATURE_NAMES)

    proba0 = _model.predict_proba(x_df)[0]
    p0 = float(proba0[pred_class])

    # SHAP local
    sv = _shap_explainer.shap_values(x_df)

    if isinstance(sv, list):
        shap_vec = np.array(sv[pred_class]).reshape(-1)
    else:
        shap_vec = np.array(sv[0, :, pred_class]).reshape(-1)

    # Top-k por |SHAP|
    order = np.argsort(np.abs(shap_vec))[::-1][:top_k]

    # Valores base
    med = X_ref.median()
    q10 = X_ref.quantile(q_low)
    q90 = X_ref.quantile(q_high)

    rows = []
    for idx in order:
        f = FEATURE_NAMES[idx]
        x_pert = x_df.copy()

        if mode == "median":
            target_val = float(med[f])

        elif mode == "aggressive":
            # Si SHAP es positivo, esa feature empuja hacia la clase -> la movemos en contra (hacia q10)
            # Si SHAP es negativo, empuja en contra -> la movemos al extremo opuesto (hacia q90)
            target_val = float(q10[f] if shap_vec[idx] > 0 else q90[f])

        else:
            raise ValueError("mode must be 'median' or 'aggressive'")

        x_pert.loc[0, f] = target_val

        proba1 = _model.predict_proba(x_pert)[0]
        p1 = float(proba1[pred_class])

        rows.append({
            "feature_perturbed": f,
            "mode": mode,
            "shap_value": float(shap_vec[idx]),
            "original_value": float(x_df.loc[0, f]),
            "perturbed_value": target_val,
            "p_class_original": p0,
            "p_class_after": p1,
            "delta_p": p1 - p0
        })

    df = pd.DataFrame(rows)
    df["abs_delta_p"] = df["delta_p"].abs()
    return df

# GLOBAL explanations 
@st.cache_data
def compute_global_explanations(_X: pd.DataFrame, _y: pd.Series, seed: int = 42, n_shap: int = 120, n_repeats: int = 20):
    # 1) SHAP global
    n = min(n_shap, len(_X))
    X_sample = _X.sample(n=n, random_state=seed)

    sv = shap_explainer.shap_values(X_sample)

    if isinstance(sv, list):
        shap_values_per_class = sv  # [class0, class1, class2]
    else:
        # array (n_samples, n_features, n_classes)
        shap_values_per_class = [sv[:, :, k] for k in range(sv.shape[2])]

    # 2) Permutation importance global
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

# para evitar recomputar global explanations cada vez
if "global_ready" not in st.session_state:
    st.session_state["global_ready"] = False
    
if not st.session_state["global_ready"]:
    with st.spinner("Inicializando explicaciones globales (una sola vez)..."):
        n_shap = len(X_bg)   
        n_repeats = 20

        X_sample, shap_values_per_class, perm_mean, perm_std = compute_global_explanations(
            X_bg, y_bg, n_shap=n_shap, n_repeats=n_repeats
        )

    st.session_state["X_sample"] = X_sample
    st.session_state["shap_values_per_class"] = shap_values_per_class
    st.session_state["perm_mean"] = perm_mean
    st.session_state["perm_std"] = perm_std
    st.session_state["global_ready"] = True

if "pred_class" not in st.session_state:
    st.session_state["pred_class"] = None

if "x_last" not in st.session_state:
    st.session_state["x_last"] = None
if "x_df_last" not in st.session_state:
    st.session_state["x_df_last"] = None

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Input")
    sepal_length = st.slider(FEATURE_LABELS[0], 0.0, 10.0, 5.0, step=0.1)
    sepal_width  = st.slider(FEATURE_LABELS[1], 0.0, 10.0, 8.7, step=0.1)
    petal_length = st.slider(FEATURE_LABELS[2], 0.0, 10.0, 4.0, step=0.1)
    petal_width  = st.slider(FEATURE_LABELS[3], 0.0, 10.0, 5.8, step=0.1)

    x = np.array([sepal_length, sepal_width, petal_length, petal_width], dtype=float)
    x_df = pd.DataFrame([x], columns=FEATURE_NAMES)

    with st.expander("Ver input"):
        st.dataframe(x_df)

with col_right:
    st.subheader("Predicción")

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

            # Guardamos clase objetivo + congelamos el input usado para explicaciones locales
            st.session_state["pred_class"] = pred
            st.session_state["x_last"] = x
            st.session_state["x_last_tuple"] = tuple(x.tolist())
            st.session_state["x_df_last"] = x_df

        except requests.exceptions.RequestException as e:
            st.error(f"No se pudo conectar con la API. Asegúrate de que está en ejecución. Error: {e}")
            st.stop()


st.divider()
st.subheader("Explainability")

pred_class = st.session_state.get("pred_class", None)

tabs = st.tabs(["Local (esta flor)", "Global (dataset)"])

with tabs[0]:
    if pred_class is None:
        st.info("Primero realiza una predicción para ver las explicaciones locales.")
    else:
         # Usamos el input congelado en la última predicción
        x_last = st.session_state["x_last"]
        x_df_last = st.session_state["x_df_last"]
        x_last_tuple = st.session_state["x_last_tuple"]
        
        c1, c2 = st.columns([1, 1], gap="large")

        with st.spinner("Calculando explicaciones locales (se cachea por input)..."):
            shap_vec, lime_html = compute_local_explanations(x_last_tuple, pred_class)
        
        with c1:
            st.markdown("### SHAP local")

            fig, ax = plt.subplots(figsize=(6, 3.6))

            order = np.argsort(np.abs(shap_vec))[::-1]
            vals = shap_vec[order]
            names = [FEATURE_LABELS[i] for i in order]

            colors = ["#ff0051" if v > 0 else "#008bfb" for v in vals]  # rojo/azul tipo SHAP

            ax.barh(names, vals, color=colors)
            ax.invert_yaxis()

            # Línea central en 0
            ax.axvline(0, color="gray", linewidth=1)

            # Limites simétricos alrededor de 0
            m = float(np.max(np.abs(vals))) if np.max(np.abs(vals)) > 0 else 1.0
            ax.set_xlim(-1.15*m, 1.15*m)

            ax.set_title(f"Contribuciones SHAP hacia: {CLASS_NAMES[pred_class]}")
            ax.set_xlabel("Impacto en el score del modelo")
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

            st.caption("Valores positivos empujan hacia la clase objetivo; negativos empujan en contra (hacia las otras clases).")

        with c2:
            st.markdown("### LIME local")
            components.html(lime_html, height=520, scrolling=True)
            st.caption("LIME ajusta un modelo interpretable local (aprox. lineal) alrededor de este punto.")

        st.markdown("### Sanity check local: perturbación (basado en SHAP)")
        st.caption("Compara una perturbación suave (mediana) frente a una perturbación agresiva (percentiles) en las variables con mayor |SHAP|.")

        top_k = st.slider("Número de variables a perturbar (top-|SHAP|)", 1, 4, 4, step=1)

        with st.spinner("Ejecutando sanity check (mediana)..."):
            df_median = sanity_check_local_perturbation_shap(
                st.session_state["x_last_tuple"],
                model_local,
                shap_explainer,
                X_bg,
                pred_class=pred_class,
                top_k=top_k,
                mode="median"
            )

        with st.spinner("Ejecutando sanity check (quantiles)..."):
            df_aggr = sanity_check_local_perturbation_shap(
                st.session_state["x_last_tuple"],
                model_local,
                shap_explainer,
                X_bg,
                pred_class=pred_class,
                top_k=top_k,
                mode="aggressive",
                q_low=0.10,
                q_high=0.90
            )

        cA, cB = st.columns(2, gap="large")

        with cA:
            st.markdown("#### Perturbación suave (mediana)")
            st.dataframe(df_median)
            st.caption("Si delta_p es casi 0, la predicción es robusta ante cambios razonables.")

        with cB:
            st.markdown("#### Perturbación agresiva (percentiles)")
            st.dataframe(df_aggr)
            st.caption("Aquí forzamos el cambio en dirección contraria a SHAP; debería afectar más a la probabilidad.")


with tabs[1]:
    
    st.markdown("### Global explanations")
    st.write("Primero mostramos **3 SHAP summary plots** (uno por clase) y luego el ranking único de **Permutation Importance**.")
    
    if not st.session_state["global_ready"]:
        with st.spinner("Calculando explicaciones globales ..."):
            n_shap =len(X_bg)
            n_repeats = 20

            X_sample, shap_values_per_class, perm_mean, perm_std = compute_global_explanations(
                X_bg, y_bg, n_shap=n_shap, n_repeats=n_repeats
            )

        st.session_state["X_sample"] = X_sample
        st.session_state["shap_values_per_class"] = shap_values_per_class
        st.session_state["perm_mean"] = perm_mean
        st.session_state["perm_std"] = perm_std
        st.session_state["global_ready"] = True
        
        st.stop()
    
    # Render desde session_state para evitar recomputar y que no se quede pillado
    X_sample = st.session_state["X_sample"]
    shap_values_per_class = st.session_state["shap_values_per_class"]
    perm_mean = st.session_state["perm_mean"]
    perm_std = st.session_state["perm_std"]

    row = st.columns(3, gap="small")
    # 1) SHAP summary por clase (3 plots)
    for k, name in enumerate(CLASS_NAMES):
        with row[k]:
            st.markdown(f"#### SHAP global – clase: **{name}**")
            fig = plt.figure(figsize=(4, 3))
            shap.summary_plot(
                shap_values_per_class[k],
                X_sample,
                feature_names=FEATURE_NAMES,
                show=False
            )
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

    st.divider()

    # 2) Permutation importance (un ranking global único)
    st.markdown("#### Permutation Feature Importance")
    st.caption("Mide la caída de accuracy al permutar cada feature.")

    left, center, right = st.columns([1, 2, 1])
    
    with center:
        fig2, ax = plt.subplots(figsize=(5, 3))
        
        order = np.argsort(perm_mean)[::-1]
        ax.barh([FEATURE_LABELS[i] for i in order], perm_mean[order], xerr=perm_std[order])
        ax.invert_yaxis()
        ax.set_xlabel("Decrease in accuracy (mean over permutations)")
        ax.set_title("Permutation Importance (test split)")
        
        st.pyplot(fig2, clear_figure=True)
        plt.close(fig2)
    
    # 3) Sanity check: feature ablation
    st.markdown("#### Sanity check global: Feature ablation (CV)")
    with st.spinner("Calculando feature ablation con validación cruzada..."):
        ablation_cv_df = feature_ablation_cv(X_bg, y_bg, seed=42, n_estimators=200, n_splits=5)

    st.dataframe(ablation_cv_df)
    st.caption("La caída media de accuracy (delta_vs_base) indica qué variables son realmente necesarias para generalizar.")
