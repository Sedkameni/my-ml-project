# main.py
from fastapi import FastAPI
import pickle
from pydantic import BaseModel
import numpy as np
from My_Model import train_and_save
import uvicorn
import os


# Initialize FastAPI
app = FastAPI()


# Request model
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Load or train model
if not os.path.exists("model.pkl"):
    print("Model file not found. Training new model...")
    train_and_save()

try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
        print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise


# Prediction endpoint
@app.post("/predict")
def predict_iris(data: IrisRequest):

    features = np.array([[data.sepal_length,
                          data.sepal_width,
                          data.petal_length,
                          data.petal_width]])

    prediction = model.predict(features)

    species = ["setosa", "versicolor", "virginica"]
    predicted_species = species[int(prediction[0])]

    return {"prediction": int(prediction[0]),
            "species": predicted_species}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)