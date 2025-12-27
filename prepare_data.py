import os
import pandas as pd
from sklearn.datasets import load_breast_cancer

# =========================
# Config
# =========================
DATA_DIR = "data"
RAW_PATH = os.path.join(DATA_DIR, "breast_cancer_raw.csv")
FINAL_PATH = os.path.join(DATA_DIR, "breast_cancer.csv")

# 6 features finales (coherentes con tu API)
FEATURE_NAMES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
]

LABEL_COL = "label"  # 0=malignant, 1=benign (sklearn)

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1) Descargar dataset desde sklearn
    ds = load_breast_cancer(as_frame=True)
    df = ds.frame.copy()  # incluye features + target

    # 2) Normalizar nombres de columnas (minúsculas)
    df.columns = [c.strip().lower() for c in df.columns]

    # 3) Renombrar target a "label"
    # sklearn lo llama normalmente "target"
    if "target" not in df.columns:
        raise ValueError("No se encontró la columna 'target' en el dataset.")
    df = df.rename(columns={"target": LABEL_COL})

    # 4) Guardar versión RAW completa (30 features + label)
    df.to_csv(RAW_PATH, index=False)

    # 5) Quedarse solo con las 6 features + label
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas esperadas en el dataset: {missing}")

    df_final = df[FEATURE_NAMES + [LABEL_COL]].copy()

    # 6) Guardar versión FINAL reducida (6 features + label)
    df_final.to_csv(FINAL_PATH, index=False)

    print("✅ Dataset descargado y preparado")
    print(f"RAW  -> {RAW_PATH} (shape={df.shape})")
    print(f"FINAL-> {FINAL_PATH} (shape={df_final.shape})")
    print("Features usadas:", FEATURE_NAMES)
    print(f"Label: '{LABEL_COL}' (0=malignant, 1=benign)")

if __name__ == "__main__":
    main()
