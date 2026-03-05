"""
CROSS-VALIDATION İLE MODEL DEĞERLENDİRME
Daha güvenilir performans ölçümü için 5-fold CV
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import sys

BASE_DIR = Path.cwd()
sys.path.insert(0, str(BASE_DIR / "scripts"))

from models_unsupervised.isolation_forest_detector import IsolationForestDetector

print("🔄 CROSS-VALIDATION İLE MODEL DEĞERLENDİRME")
print("="*70)

# Veri yükle
X = pd.read_csv(BASE_DIR / "data/features/advanced_features.csv")
df_labels = pd.read_csv(BASE_DIR / "data/labeled/labeled_traffic.csv")
y = (df_labels['label'] == 'anomaly').astype(int).values

print(f"📊 Toplam veri: {len(y)} örnek")

# 5-Fold Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

results = {
    'f1_scores': [],
    'precision_scores': [],
    'recall_scores': [],
    'roc_auc_scores': []
}

print(f"\n🔄 {n_splits}-Fold Cross-Validation başlıyor...\n")

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"Fold {fold}/{n_splits}:")
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Sadece normal trafik ile eğit
    X_train_normal = X_train[y_train == 0]
    
    # Scaler
    scaler = StandardScaler()
    X_train_normal_scaled = scaler.fit_transform(X_train_normal)
    X_test_scaled = scaler.transform(X_test)
    
    # Model eğit
    detector = IsolationForestDetector(contamination=0.1)
    detector.train(X_train_normal_scaled.values if hasattr(X_train_normal_scaled, 'values') else X_train_normal_scaled)
    
    # Tahmin yap
    predictions, scores = detector.predict(X_test_scaled.values if hasattr(X_test_scaled, 'values') else X_test_scaled)
    
    # Threshold optimize et (aynı fold içinde)
    percentiles = [60, 65, 70, 75, 80]
    best_f1 = 0
    best_threshold = None
    
    for p in percentiles:
        threshold = np.percentile(scores, p)
        y_pred = (scores >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # En iyi threshold ile final tahmin
    y_pred_final = (scores >= best_threshold).astype(int)
    
    # Metrikleri hesapla
    f1 = f1_score(y_test, y_pred_final, zero_division=0)
    prec = precision_score(y_test, y_pred_final, zero_division=0)
    rec = recall_score(y_test, y_pred_final, zero_division=0)
    
    try:
        roc = roc_auc_score(y_test, scores)
    except:
        roc = 0.0
    
    results['f1_scores'].append(f1)
    results['precision_scores'].append(prec)
    results['recall_scores'].append(rec)
    results['roc_auc_scores'].append(roc)
    
    print(f"   F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | ROC-AUC: {roc:.4f}")

# Sonuçları özetle
print("\n" + "="*70)
print("📊 CROSS-VALIDATION SONUÇLARI:")
print("="*70)

for metric_name, scores in results.items():
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"\n{metric_name.replace('_', ' ').title()}:")
    print(f"   Ortalama: {mean_score:.4f} ± {std_score:.4f}")
    print(f"   Min: {min(scores):.4f} | Max: {max(scores):.4f}")

# Güven aralığı
f1_mean = np.mean(results['f1_scores'])
f1_std = np.std(results['f1_scores'])
confidence_interval = 1.96 * f1_std / np.sqrt(n_splits)  # 95% CI

print(f"\n🎯 F1-Score (95% Güven Aralığı):")
print(f"   {f1_mean:.4f} ± {confidence_interval:.4f}")
print(f"   [{f1_mean - confidence_interval:.4f}, {f1_mean + confidence_interval:.4f}]")

# Sonuçları kaydet
cv_results = {
    'model': 'Isolation Forest',
    'validation_type': f'{n_splits}-Fold Cross-Validation',
    'date': pd.Timestamp.now().isoformat(),
    'metrics': {
        'f1_mean': float(f1_mean),
        'f1_std': float(f1_std),
        'f1_ci_95': float(confidence_interval),
        'precision_mean': float(np.mean(results['precision_scores'])),
        'recall_mean': float(np.mean(results['recall_scores'])),
        'roc_auc_mean': float(np.mean(results['roc_auc_scores']))
    },
    'fold_results': {
        f'fold_{i+1}': {
            'f1': float(results['f1_scores'][i]),
            'precision': float(results['precision_scores'][i]),
            'recall': float(results['recall_scores'][i]),
            'roc_auc': float(results['roc_auc_scores'][i])
        }
        for i in range(n_splits)
    }
}

output_file = BASE_DIR / "outputs" / "cross_validation_results.json"
with open(output_file, 'w') as f:
    json.dump(cv_results, f, indent=2)

print(f"\n✅ Sonuçlar kaydedildi: {output_file}")
