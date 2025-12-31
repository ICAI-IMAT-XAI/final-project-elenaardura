[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/d89f4r04)

# 🧠 Iris Classification with Explainable AI (XAI)

Este repositorio contiene un proyecto de **clasificación multiclase** basado en el dataset Iris, junto con una **interfaz web interactiva** que permite analizar el comportamiento del modelo mediante técnicas de **Explainable Artificial Intelligence (XAI)** a nivel global y local.

El sistema se ejecuta **en local** utilizando **Docker**, combinando una API de predicción con una aplicación web desarrollada en Streamlit para la visualización de resultados y explicaciones.

---

## Estructura del proyecto

- app.py: API de predicción
- app_web.py: Interfaz web (Streamlit)
- train.py: Entrenamiento del modelo
- model.pkl: Modelo entrenado (se genera con train.py)
-  data/iris_dataset.csv: Dataset Iris (se genera con train.py si no existe)
- Dockerfile: Dockerfile de la API
- Dockerfile.web: Dockerfile de la app web
- docker-compose.yml: Orquestación de servicios
- requirements.txt
- README.md
- PracticaFinal_Elena_Ardura.pdf: Informe del proyecto


## Paso previo obligatorio: entrenamiento del modelo

Antes de ejecutar el proyecto con Docker, es **imprescindible entrenar el modelo**.

El entrenamiento **no se realiza automáticamente** al levantar los contenedores, por lo que debe ejecutarse previamente:

```bash
python train.py
```

Este script realizará las siguientes acciones:
1. Genera el dataset Iris si no existe
2. Entrena un modelo Random Forest
3. Guarda el modelo entrenado en model.pkl
4. Genera la matriz de confusión del conjunto de test

Sin este paso, la aplicación no funcionará, ya que la API y la interfaz web dependen del archivo model.pkl.



▶️ Ejecución del proyecto
Una vez entrenado el modelo, ejecutar:

```bash
docker-compose up
```

Las imágenes Docker del proyecto ya están publicadas en Docker Hub, por lo que no es necesario construirlas localmente. Al ejecutar docker-compose, Docker descarga automáticamente las imágenes desde Docker Hub.

Si no funciona la descarga automática, puede construirse la imagen localmente antes de ejecutar el docker compose anterior:

```bash
docker build -t elenaardura/practica-final-xai-api -f Dockerfile .
docker build -t elenaardura/practica-final-xai-web -f Dockerfile.web .
```

Tras levantar los contenedores, estarán disponibles los siguientes servicios:

- API de predicción: http://localhost:5000
- Interfaz web (Streamlit): http://localhost:8501

Desde la interfaz web se pueden:
- Introducir valores de entrada
- Obtener predicciones del modelo
- Analizar explicaciones globales y locales
- Explorar sanity checks y perturbaciones de variables

## Técnicas de explicabilidad incluidas

- SHAP global (análisis por clase en un problema multiclase)
- Permutation Feature Importance
- SHAP local
- LIME local

Sanity checks:
- Ablación de variables
- Perturbaciones suaves y agresivas guiadas por SHAP