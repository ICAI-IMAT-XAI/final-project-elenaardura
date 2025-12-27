import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import json

def train_model(C):
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)

    # =========================
    # 1. Cargar dataset local
    # =========================
    df = pd.read_csv("data/breast_cancer.csv")

    X = df.drop(columns=["label"])
    print(X.shape)
    print(X.columns)
    y = df["label"]

    with mlflow.start_run():
        # =========================
        # 2. Train / Test split
        # =========================

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
            stratify=y  # importante en binaria
        )

        # =========================
        # 3. Pipeline: Scaler + SVC
        # =========================
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("svc", SVC(
                    kernel="rbf",
                    C=C,
                    gamma="scale",
                    probability=True,
                    random_state=42
                ))
            ]
        )

        # =========================
        # 4. Entrenamiento
        # =========================
        model.fit(X_train, y_train)

        # =========================
        # 5. Evaluación básica
        # =========================
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # =========================
        # 6. Guardar modelo
        # =========================
        joblib.dump(model, "model.pkl")

        # Registrar el modelo con MLflow
        mlflow.sklearn.log_model(model, "svc-model")

        # Registrar parámetros y métricas
        mlflow.log_param("model_type", "SVC")
        mlflow.log_param("kernel", "rbf")
        mlflow.log_param("C", C)
        mlflow.log_param("gamma", "scale")
        mlflow.log_param("probability", True)
        mlflow.log_param("test_size", 0.3)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("accuracy", accuracy)

        print(f"Modelo entrenado y precisión: {accuracy:.4f}")
        print("Experimento registrado con MLflow.")
        
        # 1. Generar la matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicciones')
        plt.ylabel('Valores Reales')
        plt.savefig('confusion_matrix.png')
        print("Matriz de confusión guardada como 'confusion_matrix.png'")
        
        # Guardar el artefacto en MLflow remoto
        mlflow.log_artifact("confusion_matrix.png")
        metrics = {
            "accuracy": accuracy
        }
        
        with open("mlflow_metrics.json", "w") as f:
            json.dump(metrics, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()
    train_model(args.C)