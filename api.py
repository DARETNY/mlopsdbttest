from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from fastapi.responses import RedirectResponse

# 1. Model Yükleme
model = joblib.load("model.pkl")

# 2. Kurumsal Metadata
app = FastAPI(
    title="Şarap Kalitesi Tahmin Motoru",
    description="Canlı sistemlerden gelen verilerle şarap kalite sınıfını tahmin eden kurumsal API.",
    version="1.0.0",
    contact={
        "name": "Veri Bilimi ve MLOps Ekibi",
        "email": "mlops@sirket.com",
    }
)


# 3. Yönlendirme (Ana sayfadan dokümantasyona)
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/redoc")


# 4. Veri Validasyon Şeması (Wine veri setindeki 13 özellik)
class WinePredictionRequest(BaseModel):
    alcohol: float = Field(..., description="Alkol oranı")
    malic_acid: float = Field(..., description="Malik asit miktarı")
    ash: float = Field(..., description="Kül oranı")
    alcalinity_of_ash: float = Field(..., description="Kül alkalinitesi")
    magnesium: float = Field(..., description="Magnezyum miktarı")
    total_phenols: float = Field(..., description="Toplam fenoller")
    flavanoids: float = Field(..., description="Flavonoidler")
    nonflavanoid_phenols: float = Field(..., description="Flavonoid olmayan fenoller")
    proanthocyanins: float = Field(..., description="Proantosiyaninler")
    color_intensity: float = Field(..., description="Renk yoğunluğu")
    hue: float = Field(..., description="Renk tonu (Hue)")
    od280_od315_of_diluted_wines: float = Field(..., description="Seyreltilmiş şarapların OD280/OD315 değeri")
    proline: float = Field(..., description="Prolin miktarı")

    # Arayüz için kurumsal örnek veri
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "alcohol": 14.23,
                    "malic_acid": 1.71,
                    "ash": 2.43,
                    "alcalinity_of_ash": 15.6,
                    "magnesium": 127.0,
                    "total_phenols": 2.8,
                    "flavanoids": 3.06,
                    "nonflavanoid_phenols": 0.28,
                    "proanthocyanins": 2.29,
                    "color_intensity": 5.64,
                    "hue": 1.04,
                    "od280_od315_of_diluted_wines": 3.92,
                    "proline": 1065.0
                }
            ]
        }
    }


# 5. Tahmin Endpoint'i
@app.post("/predict", tags=["Tahmin İşlemleri"], summary="Şarap kalitesini sınıflandır")
def predict(data: WinePredictionRequest):
    # Gelen veriyi DataFrame'e çevir
    df = pd.DataFrame([data.model_dump()])

    # Model tahmini
    prediction = model.predict(df)

    return {"tahmin_edilen_sinif": int(prediction[0])}
