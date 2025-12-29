import joblib
from flask import Flask, request, jsonify
import numpy as np

MODEL_PATH = "model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Modelo no cargado. Ejecuta train.py para generar model.pkl"}), 500

    try:
        data = request.get_json(force=True)
        features = np.array(data["features"], dtype=float).reshape(1, -1)

        pred = int(model.predict(features)[0])
        proba = model.predict_proba(features)[0].tolist()

        species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        return jsonify({
            "prediction": pred,
            "species": species_map.get(pred, "unknown"),
            "probability": {
                "setosa": proba[0],
                "versicolor": proba[1],
                "virginica": proba[2],
            },
            "confidence": float(max(proba))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    print("API Iris en http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)