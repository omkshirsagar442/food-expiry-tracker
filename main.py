from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI()

model = joblib.load("fruit_quality_model.pkl")
fruit_encoder = joblib.load("fruit_encoder.pkl")
class_encoder = joblib.load("class_encoder.pkl")

class SensorData(BaseModel):
    fruit: str
    temp: float
    humidity: float
    moisture: float
    weight: float

@app.get("/")
def home():
    return {"message": "API Running"}

@app.post("/predict")
def predict(data: SensorData):

    fruit_encoded = fruit_encoder.transform([data.fruit])[0]

    input_data = pd.DataFrame({
        "Fruit": [fruit_encoded],
        "Temp": [data.temp],
        "Humid (%)": [data.humidity]
    })

    prediction = model.predict(input_data)
    result = class_encoder.inverse_transform(prediction)[0].capitalize()

    return {"prediction": result}


# 🔥 REQUIRED FOR RENDER
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)