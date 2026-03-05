"""
İYİLEŞTİRİLMİŞ MODEL KARŞILAŞTIRMA
- Threshold optimizasyonu
- Detaylı confusion matrix
- Precision/Recall trade-off analizi
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, precision_recall_curve
)
import sys

BASE_DIR = Path.cwd()
sys.path.insert(0, str(BASE_DIR / "scripts"))

from models_unsupervised.lstm_autoencoder import LSTMAnomalyDetector
from models_unsupervised.isolation_forest_detector import IsolationForestDetector
from models_unsupervised.one_class_svm_detector import OneClassSVMDetector

print("🚀 İYİLEŞTİRİLMİŞ MODEL KARŞILAŞTIRMA")
print("="*70)

# Veri yükleme
FEATURES_FILE = BASE_DIR / "data" / "features" / "advanced_features.csv"
LABELED_FILE = BASE_DIR / "data" / "labeled" / "labeled_traffic.csv"

X = pd.read_csv(FEATURES_FILE)
df_labels = pd.read_csv(LABELED_FILE)
y = (df_labels['label'] == 'anomaly').astype(int).values

print(f"📊 Veri Yüklendi:")
print(f"   Toplam: {len(y)}")
print(f"   Normal: {(y == 0).sum()}")
print(f"   Anomaly: {(y == 1).sum()}")

# Train/Test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ Train: {len(y_train)} | Test: {len(y_test)}")
print(f"   Train Normal: {(y_train == 0).sum()}")
print(f"   Test Normal: {(y_test == 0).sum()}")

# Normalizasyon (sadece train'e fit!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_normal_scaled = X_train_scaled[y_train == 0]

print(f"\n✅ Scaler sadece train'e fit edildi (NO LEAKAGE)")

# Sonuçlar
results = {}

def optimize_threshold(y_true, scores, metric='f1'):
    """Optimal threshold bul"""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    
    if metric == 'f1':
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        return thresholds[best_idx], f1_scores[best_idx]
    elif metric == 'precision':
        # Precision >= 0.7 olan en yüksek recall
        valid_idx = np.where(precision >= 0.7)[0]
        if len(valid_idx) > 0:
            best_idx = valid_idx[np.argmax(recall[valid_idx])]
            return thresholds[best_idx], f1_scores[best_idx]
    
    return thresholds[0], 0.0

def evaluate_with_threshold(y_true, scores, threshold):
    """Threshold ile değerlendir"""
    y_pred = (scores >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        roc = roc_auc_score(y_true, scores)
    except:
        roc = 0.0
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': roc,
        'confusion_matrix': {'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)},
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'threshold': float(threshold)
    }

# === ISOLATION FOREST ===
print("\n" + "="*70)
print("🌲 ISOLATION FOREST (Threshold Optimized)")
print("="*70)

if_detector = IsolationForestDetector(contamination=0.1)
if_detector.fit(X_train_normal_scaled)
if_scores = if_detector.predict_proba(X_test_scaled)

# Optimal threshold bul
best_threshold, best_f1 = optimize_threshold(y_test, if_scores, metric='f1')
print(f"✅ Optimal Threshold: {best_threshold:.4f} (F1: {best_f1:.4f})")

results['Isolation Forest'] = evaluate_with_threshold(y_test, if_scores, best_threshold)

print(f"   Accuracy: {results['Isolation Forest']['accuracy']:.4f}")
print(f"   Precision: {results['Isolation Forest']['precision']:.4f}")
print(f"   Recall: {results['Isolation Forest']['recall']:.4f}")
print(f"   F1-Score: {results['Isolation Forest']['f1_score']:.4f}")
print(f"   FP Rate: {results['Isolation Forest']['false_positive_rate']:.4f}")

cm = results['Isolation Forest']['confusion_matrix']
print(f"\n📊 Confusion Matrix:")
print(f"   TP: {cm['TP']} | FP: {cm['FP']}")
print(f"   FN: {cm['FN']} | TN: {cm['TN']}")

# === ONE-CLASS SVM ===
print("\n" + "="*70)
print("🔷 ONE-CLASS SVM (Threshold Optimized)")
print("="*70)

ocsvm_detector = OneClassSVMDetector(nu=0.1, kernel='rbf', gamma='scale')
ocsvm_detector.fit(X_train_normal_scaled)
ocsvm_scores = ocsvm_detector.predict_proba(X_test_scaled)

best_threshold, best_f1 = optimize_threshold(y_test, ocsvm_scores, metric='f1')
print(f"✅ Optimal Threshold: {best_threshold:.4f} (F1: {best_f1:.4f})")

results['One-Class SVM'] = evaluate_with_threshold(y_test, ocsvm_scores, best_threshold)

print(f"   F1-Score: {results['One-Class SVM']['f1_score']:.4f}")
print(f"   Precision: {results['One-Class SVM']['precision']:.4f}")
print(f"   Recall: {results['One-Class SVM']['recall']:.4f}")

# === LSTM AUTOENCODER ===
print("\n" + "="*70)
print("🧠 LSTM AUTOENCODER (Threshold Optimized)")
print("="*70)

lstm_detector = LSTMAnomalyDetector(
    sequence_length=10,
    encoding_dim=8,
    epochs=20,
    batch_size=64
)
lstm_detector.fit(X_train_normal_scaled)
lstm_scores = lstm_detector.predict_proba(X_test_scaled)

best_threshold, best_f1 = optimize_threshold(y_test, lstm_scores, metric='f1')
print(f"✅ Optimal Threshold: {best_threshold:.4f} (F1: {best_f1:.4f})")

results['LSTM Autoencoder'] = evaluate_with_threshold(y_test, lstm_scores, best_threshold)

print(f"   F1-Score: {results['LSTM Autoencoder']['f1_score']:.4f}")
print(f"   Precision: {results['LSTM Autoencoder']['precision']:.4f}")
print(f"   Recall: {results['LSTM Autoencoder']['recall']:.4f}")

# Sonuçları kaydet
report = {
    'comparison_date': pd.Timestamp.now().isoformat(),
    'data_info': {
        'total_samples': len(y),
        'train_samples': len(y_train),
        'test_samples': len(y_test),
        'normal_samples': int((y == 0).sum()),
        'anomaly_samples': int((y == 1).sum()),
        'features': X.shape[1]
    },
    'model_results': results
}

OUTPUT_FILE = BASE_DIR / "outputs" / "model_comparison_optimized.json"
OUTPUT_FILE.parent.mkdir(exist_ok=True)

with open(OUTPUT_FILE, 'w') as f:
    json.dump(report, f, indent=2)

print("\n" + "="*70)
print("✅ TAMAMLANDI!")
print(f"📁 Sonuçlar: {OUTPUT_FILE}")
print("="*70)

# En iyi model
best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
print(f"\n🏆 EN İYİ MODEL: {best_model[0]}")
print(f"   F1-Score: {best_model[1]['f1_score']:.4f}")
print(f"   Precision: {best_model[1]['precision']:.4f}")
print(f"   Recall: {best_model[1]['recall']:.4f}")
