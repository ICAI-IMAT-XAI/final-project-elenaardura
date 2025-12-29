import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

DATA_PATH = "data/iris_dataset.csv"
MODEL_PATH = "model.pkl"

def ensure_data_dir():
    os.makedirs("data", exist_ok=True)

def build_dataset_csv():
    """Crea data/iris_dataset.csv desde sklearn Iris para que el proyecto sea reproducible."""
    ensure_data_dir()
    iris = load_iris(as_frame=True)
    df = iris.frame  # incluye target
    df.to_csv(DATA_PATH, index=False)
    return df

def train_model(n_estimators: int):
    # 1) Dataset
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        df = build_dataset_csv()

    X = df.drop(columns=["target"])
    y = df["target"]

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 3) Modelo
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # 4) Evaluación
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy test: {acc:.4f}")

    # 5) Guardar modelo
    joblib.dump(model, MODEL_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")

    # 6) Matriz de confusión (opcional)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Iris)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Matriz de confusión guardada como 'confusion_matrix.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()
    train_model(args.n_estimators)
