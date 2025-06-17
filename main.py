from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Load your saved models
lr_model = joblib.load("logistic_regression_model.pkl")
dt_model = joblib.load("decision_tree_model.pkl")
kmeans_model = joblib.load("kmeans_model.pkl")

# Define input schema for request body
class TripFeatures(BaseModel):
    passenger_count: int
    trip_distance: float
    fare_amount: float
    # add other features as needed...

@app.post("/predict_payment/")
def predict_payment(data: TripFeatures):
    features = np.array([[data.passenger_count, data.trip_distance, data.fare_amount]])
    lr_pred = lr_model.predict(features)[0]
    dt_pred = dt_model.predict(features)[0]
    return {"logistic_regression": int(lr_pred), "decision_tree": int(dt_pred)}

@app.post("/predict_cluster/")
def predict_cluster(data: TripFeatures):
    features = np.array([[data.passenger_count, data.trip_distance, data.fare_amount]])
    cluster_pred = kmeans_model.predict(features)[0]
    return {"cluster": int(cluster_pred)}
