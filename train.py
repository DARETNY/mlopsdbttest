import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Örnek Veri
X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
y = [0, 1, 0]

# Eğitim
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# Modeli diske kaydetme (Artifact creation)
joblib.dump(model, "model.pkl")
print("Model eğitildi ve model.pkl olarak başarıyla kaydedildi.")
