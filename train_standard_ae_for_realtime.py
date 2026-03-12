from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scripts.models_unsupervised.standard_autoencoder import StandardAutoencoder

BASE_DIR = Path(__file__).resolve().parent

# 1) Veriyi yükle
X = pd.read_csv(BASE_DIR / "data" / "features" / "advanced_features.csv")
df_labels = pd.read_csv(BASE_DIR / "data" / "labeled" / "labeled_traffic.csv")

y = (df_labels["label"] == "anomaly").astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Sadece normal trafikten eğitim
X_train_normal = X_train[y_train == 0]

# Scale
scaler = StandardScaler()
X_train_normal_scaled = scaler.fit_transform(X_train_normal)

# 2) Modeli kur ve eğit
ae = StandardAutoencoder(encoding_dim=16)
ae.build_model(input_dim=X.shape[1])

print("🧠 Standard Autoencoder eğitiliyor (realtime için)...")
ae.train(
    X_train_normal.values,  # sınıf içi scaler kullanmak istersen X_train_normal_scaled yerine bunu kullan
    epochs=20,
    batch_size=64,
    patience=5,
    verbose=1,
)

# 3) Modeli realtime NIDS için kaydet
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

ae.save_model(
    model_path=MODELS_DIR / "standard_autoencoder.keras",
    scaler_path=MODELS_DIR / "standard_ae_scaler.pkl",
)

print("✅ Realtime için Standard Autoencoder model dosyaları üretildi.")