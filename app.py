import joblib
from flask import Flask, request, jsonify, Response
import numpy as np
import pandas as pd
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

# Breast cancer tiene 30 features
N_FEATURES = 6
FEATURE_NAMES = ["mean radius","mean texture","mean perimeter","mean area","mean smoothness","mean compactness"]

PREDICTION_COUNTER = Counter(
    "breast_cancer_prediction_count",
    "Contador de predicciones del modelo Breast Cancer por clase",
    ["label"],  # "benign" o "malignant"
)

# Cargar el modelo entrenado
try:
    model = joblib.load('model.pkl')
except FileNotFoundError:
    print("Error: 'model.pkl' no encontrado. Por favor, asegúrate de haber ejecutado el script de entrenamiento.")
    model = None

# Inicializar la aplicación Flask
app = Flask(__name__)

@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Modelo no cargado. Entrena el modelo primero."}), 500

    try:
        data = request.get_json(force=True)
        
        # features = np.array(data["features"], dtype=float).reshape(1, -1)
        features = pd.DataFrame([data["features"]], columns=FEATURE_NAMES)
    
        # Predicción (0=malignant, 1=benign en sklearn)
        pred = int(model.predict(features)[0])

        # Probabilidades (porque entrenaste con probability=True)
        proba = model.predict_proba(features)[0]  # [P(clase 0), P(clase 1)]

        # Etiquetas humanas
        pred_label = "benign" if pred == 1 else "malignant"
        prob_malignant = float(proba[0])
        prob_benign = float(proba[1])

        PREDICTION_COUNTER.labels(label=pred_label).inc()

        return jsonify({
            "prediction": pred,
            "prediction_label": pred_label,
            "probability": {
                "malignant": prob_malignant,
                "benign": prob_benign
            },
            "confidence": float(max(proba))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)