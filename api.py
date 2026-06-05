from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. API ayağa kalkarken modeli belleğe yükle
model = joblib.load("model.pkl")

app = FastAPI(title="Kurumsal ML API")


# 2. Dışarıdan beklediğimiz verinin formatı (Güvenlik ve validasyon için)
class PredictionRequest(BaseModel):
    feature1: float
    feature2: float


# 3. Tahmin yapacak Endpoint
@app.post("/predict")
def predict(data: PredictionRequest):
    # Gelen JSON verisini DataFrame'e çevir
    df = pd.DataFrame([data.model_dump()])

    # Modele sor ve sonucu al
    prediction = model.predict(df)

    return {"tahmin_edilen_sinif": int(prediction[0])}
