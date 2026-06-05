import mlflow.sklearn
from mlflow.models import infer_signature  # 1. YENİ: İmza kütüphanesini ekle
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

mlflow.set_experiment("Sarap_Kalitesi_Tahmini")

with mlflow.start_run():
    n_estimators = 100
    max_depth = 10

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # 2. YENİ: Sisteme giren ve çıkan veriyi gösterip imza (şema) çıkarıyoruz
    signature = infer_signature(X_test, predictions)

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)

    # 3. DEĞİŞİKLİK: Modeli kaydederken imzayı (signature) da dahil ediyoruz
    mlflow.sklearn.log_model(model, "random_forest_model", signature=signature, registered_model_name=""
                                                                                                      "Sarap_Kalitesi_Modeli")

    print("Model başarıyla ve sektör standardı bir İMZA ile kaydedildi!")
