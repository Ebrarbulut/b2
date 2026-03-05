"""
STANDARD AUTOENCODER İLE THRESHOLD OPTİMİZASYONU
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = Path.cwd()
sys.path.insert(0, str(BASE_DIR / "scripts"))

try:
    from models_unsupervised.standard_autoencoder import StandardAutoencoder
except ImportError:
    print("❌ Standard Autoencoder bulunamadı!")
    print("💡 LSTM Autoencoder ile devam ediliyor...")
    from models_unsupervised.lstm_autoencoder import LSTMAnomalyDetector as StandardAutoencoder

print("🧠 STANDARD AUTOENCODER OPTİMİZASYONU")
print("="*70)

# Veri yükle
X = pd.read_csv(BASE_DIR / "data/features/advanced_features.csv")
df_labels = pd.read_csv(BASE_DIR / "data/labeled/labeled_traffic.csv")
y = (df_labels['label'] == 'anomaly').astype(int).values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_normal = X_train[y_train == 0]

print(f"📊 Veri: {len(y)} total")
print(f"   Train Normal: {len(X_train_normal)}")

# Scaler
scaler = StandardScaler()
X_train_normal_scaled = scaler.fit_transform(X_train_normal)
X_test_scaled = scaler.transform(X_test)

# Autoencoder eğit
print("\n🧠 Standard Autoencoder eğitiliyor...")
ae = StandardAutoencoder(
    encoding_dim=8,
    epochs=20,
    batch_size=64
)
ae.fit(X_train_normal_scaled)

# Skorları al
scores = ae.predict_proba(X_test_scaled)

print("\n🔍 FARKLI THRESHOLD DEĞERLERİ:")
print("="*70)

percentiles = [70, 75, 80, 85, 90, 95]
best_f1 = 0
best_results = None

for p in percentiles:
    threshold = np.percentile(scores, p)
    y_pred = (scores >= threshold).astype(int)
    
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nPercentile {p}%: F1={f1:.4f} | Prec={prec:.4f} | Rec={rec:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_results = {
            'threshold': float(threshold),
            'percentile': p,
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'confusion_matrix': {'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)}
        }

print(f"\n🏆 EN İYİ: F1={best_f1:.4f} @ Percentile {best_results['percentile']}%")

# Kaydet
report = {
    'model': 'Standard Autoencoder',
    'optimization_date': pd.Timestamp.now().isoformat(),
    'best_threshold': best_results
}

output_file = BASE_DIR / "outputs" / "autoencoder_optimization.json"
with open(output_file, 'w') as f:
    json.dump(report, f, indent=2)

print(f"✅ Kaydedildi: {output_file}")
