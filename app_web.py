import os
import streamlit as st
import requests

# =========================
# Config
# =========================
API_URL = os.environ.get("API_URL")

FEATURE_LABELS = [
    "Mean radius",
    "Mean texture",
    "Mean perimeter",
    "Mean area",
    "Mean smoothness",
    "Mean compactness",
]

st.title("API de Predicción - Breast Cancer (SVC)")
st.write("Introduce las 6 características del tumor y obtén una predicción **benign** vs **malignant**.")

# Inputs
mean_radius = st.slider(FEATURE_LABELS[0], 5.0, 30.0, 14.5, step=0.1)
mean_texture = st.slider(FEATURE_LABELS[1], 5.0, 40.0, 20.2, step=0.1)
mean_perimeter = st.slider(FEATURE_LABELS[2], 40.0, 200.0, 95.3, step=0.1)
mean_area = st.slider(FEATURE_LABELS[3], 100.0, 2500.0, 650.0, step=1.0)
mean_smoothness = st.slider(FEATURE_LABELS[4], 0.02, 0.30, 0.102, step=0.001)
mean_compactness = st.slider(FEATURE_LABELS[5], 0.00, 0.50, 0.118, step=0.001)

if st.button("Obtener Predicción"):
    features = [
        mean_radius,
        mean_texture,
        mean_perimeter,
        mean_area,
        mean_smoothness,
        mean_compactness,
    ]
    payload = {"features": features}

    with st.expander("Debug request"):
        st.code(f"POST {API_URL}")
        st.json(payload)

    try:
        st.write("Enviando petición a la API: ", API_URL)
        response = requests.post(API_URL, json=payload, timeout=10)

        content_type = response.headers.get("Content-Type", "")

        # Si no es 200, muestra info útil
        if response.status_code != 200:
            st.error(f"Error HTTP {response.status_code}")
            st.write(f"Content-Type: `{content_type}`")

            # Muestra un preview (HTML o lo que sea) sin petar la app
            preview = response.text[:800]
            st.code(preview)
            st.stop()

        # Si es 200 pero te devuelve HTML, lo detectamos
        if "application/json" not in content_type.lower():
            st.error("La API respondió 200 pero no devolvió JSON (parece HTML u otro formato).")
            st.write(f"Content-Type: `{content_type}`")
            st.code(response.text[:800])
            st.stop()

        # ✅ parsear JSON seguro
        result = response.json()

        pred_label = result.get("prediction_label")
        pred = result.get("prediction")
        confidence = result.get("confidence")
        proba = result.get("probability", {})

        if pred_label is not None:
            st.success(f"La predicción es: **{str(pred_label).upper()}** (clase={pred})")
        else:
            label_map = {0: "MALIGNANT", 1: "BENIGN"}
            st.success(f"La predicción es: **{label_map.get(pred, 'DESCONOCIDA')}** (clase={pred})")

        if confidence is not None:
            st.write(f"Confianza (máxima probabilidad): **{float(confidence):.3f}**")

        # if isinstance(proba, dict) and proba:
        #     st.write("Probabilidades:")
        #     st.json(proba)

    except requests.exceptions.RequestException as e:
        st.error(f"No se pudo conectar con la API en {API_URL}. Error: {e}")
