from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="API de Clasificación de Flores Iris",
    description="API dockerizada para predecir la especie de una flor Iris.",
    version="1.0"
)

#Cargar el modelo guardado
model = joblib.load("model.pkl")

#Mapeo de predicciones a nombres de especies
TARGET_NAMES = ["setosa", "versicolor", "virginica"]

#Estructura del JSON de entrada
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "API de Machine Learning activa y lista para recibir peticiones en /docs"}

@app.post("/predict")
def predict(data: IrisInput):
    #Convertir los datos a un arreglo compatible con el modelo
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    
    #Hacer la predicción
    prediction = model.predict(features)[0]
    species = TARGET_NAMES[prediction]
    
    return {
        "prediction_class": int(prediction),
        "species": species
    }