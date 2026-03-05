"""
BASIT THRESHOLD OPTİMİZASYONU
F1-Score'u artırmak için threshold ayarla
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

BASE_DIR = Path.cwd()
sys.path.insert(0, str(BASE_DIR / "scripts"))

from models_unsupervised.isolation_forest_detector import IsolationForestDetector

print("🎯 THRESHOLD OPTİMİZASYONU")
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

print(f"📊 Veri: {len(y)} total | {(y==0).sum()} normal | {(y==1).sum()} anomaly")
print(f"   Train: {len(y_train)} | Test: {len(y_test)}")

# Isolation Forest eğit
print("\n🌲 Isolation Forest eğitiliyor...")
detector = IsolationForestDetector(contamination=0.1)
detector.train(X_train_normal)

# Skorları al
predictions, scores = detector.predict(X_test)

print("\n🔍 FARKLI THRESHOLD DEĞERLERİ:")
print("="*70)

# Farklı threshold'ları dene
percentiles = [50, 60, 70, 75, 80, 85, 90, 95]
best_f1 = 0
best_threshold = None
best_results = None

for p in percentiles:
    threshold = np.percentile(scores, p)
    y_pred = (scores >= threshold).astype(int)
    
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nPercentile {p}% (Threshold: {threshold:.4f}):")
    print(f"   Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(f"   TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print(f"   FP Rate: {fp/(fp+tn):.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_results = {
            'threshold': float(threshold),
            'percentile': p,
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'confusion_matrix': {'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)},
            'false_positive_rate': float(fp/(fp+tn)) if (fp+tn) > 0 else 0
        }

print("\n" + "="*70)
print(f"🏆 EN İYİ THRESHOLD: Percentile {best_results['percentile']}%")
print(f"   Threshold Value: {best_threshold:.4f}")
print(f"   F1-Score: {best_f1:.4f}")
print(f"   Precision: {best_results['precision']:.4f}")
print(f"   Recall: {best_results['recall']:.4f}")
print(f"   FP Rate: {best_results['false_positive_rate']:.4f}")

# Sonuçları kaydet
report = {
    'model': 'Isolation Forest',
    'optimization_date': pd.Timestamp.now().isoformat(),
    'data_info': {
        'total_samples': len(y),
        'normal_samples': int((y == 0).sum()),
        'anomaly_samples': int((y == 1).sum())
    },
    'best_threshold': best_results
}

output_file = BASE_DIR / "outputs" / "threshold_optimization.json"
with open(output_file, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n✅ Sonuçlar kaydedildi: {output_file}")
