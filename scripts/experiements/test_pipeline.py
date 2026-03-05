"""
🧪 COMPLETE PIPELINE TEST - HAZIR VERİ SETLERİ İLE TEST
========================================================

Bu script, hazır veri setlerinde tüm pipeline'ı test eder:
1. Veri yükleme ve etiketleme
2. Feature engineering
3. Model eğitimi
4. Threshold optimizasyonu
5. Değerlendirme ve raporlama

KULLANIM:
    python scripts/test_pipeline.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Script dizinini path'e ekle
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

try:
    from lstm_autoencoder import LSTMAnomalyDetector
    from threshold_optimizer import ThresholdOptimizer
except ImportError as e:
    print(f"❌ Import hatası: {e}")
    print("💡 Script'lerin doğru dizinde olduğundan emin olun.")
    sys.exit(1)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("🧪 COMPLETE PIPELINE TEST - HAZIR VERİ SETLERİ")
print("=" * 70)

# === PATHS ===
DATA_DIR = BASE_DIR / "data"
LABELED_FILE = DATA_DIR / "labeled" / "labeled_traffic.csv"
FEATURES_FILE = DATA_DIR / "features" / "advanced_features.csv"
MODELS_DIR = BASE_DIR / "models" / "autoencoder"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# === ADIM 1: VERİ YÜKLEME VE KONTROL ===
print("\n" + "="*70)
print("📂 ADIM 1: VERİ YÜKLEME")
print("="*70)

if not LABELED_FILE.exists():
    print(f"❌ {LABELED_FILE} bulunamadı!")
    print("💡 Önce 'python scripts/label_traffic.py' çalıştırın.")
    sys.exit(1)

df = pd.read_csv(LABELED_FILE)
print(f"✅ {len(df)} kayıt yüklendi")

# Label kontrolü
if 'label' not in df.columns:
    print("❌ 'label' sütunu bulunamadı!")
    sys.exit(1)

print(f"\n📊 Label Dağılımı:")
print(df['label'].value_counts())

# === ADIM 2: FEATURE ENGINEERING ===
print("\n" + "="*70)
print("🔧 ADIM 2: FEATURE ENGINEERING")
print("="*70)

if FEATURES_FILE.exists():
    print(f"✅ Mevcut feature dosyası kullanılıyor: {FEATURES_FILE}")
    X = pd.read_csv(FEATURES_FILE)
else:
    print("⚠️ Feature dosyası yok, oluşturuluyor...")
    import subprocess
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "scripts" / "advanced_features.py")],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        X = pd.read_csv(FEATURES_FILE)
        print(f"✅ Feature engineering tamamlandı: {X.shape}")
    else:
        print(f"❌ Feature engineering hatası:\n{result.stderr}")
        sys.exit(1)

# Label'ları al
if 'label' in df.columns:
    y = (df['label'] == 'anomaly').astype(int)
else:
    print("❌ Label bilgisi bulunamadı!")
    sys.exit(1)

# Feature ve label sayılarını eşitle
min_len = min(len(X), len(y))
X = X.iloc[:min_len]
y = y.iloc[:min_len]

print(f"✅ Feature matrix: {X.shape}")
print(f"✅ Label vector: {y.shape}")
print(f"   - Normal: {(y == 0).sum()}")
print(f"   - Anomaly: {(y == 1).sum()}")

# === ADIM 3: VERİ HAZIRLAMA ===
print("\n" + "="*70)
print("📊 ADIM 3: VERİ HAZIRLAMA")
print("="*70)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train set: {X_train.shape[0]} örnek")
print(f"✅ Test set: {X_test.shape[0]} örnek")

# Normalizasyon
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Veri normalizasyonu tamamlandı")

# === ADIM 4: MODEL EĞİTİMİ ===
print("\n" + "="*70)
print("🧠 ADIM 4: LSTM AUTOENCODER EĞİTİMİ")
print("="*70)

# Sadece normal trafik ile eğit
X_train_normal = X_train_scaled[y_train == 0]
print(f"📊 Normal trafik ile eğitim: {X_train_normal.shape[0]} örnek")

# LSTM parametreleri
sequence_length = 10
n_features = X_train_scaled.shape[1]

print(f"📐 Sequence length: {sequence_length}")
print(f"📐 Feature sayısı: {n_features}")

# Detector oluştur
detector = LSTMAnomalyDetector(
    sequence_length=sequence_length,
    n_features=n_features,
    latent_dim=16
)

# Model oluştur
detector.build_model(lstm_units=[64, 32])
print("✅ Model yapısı oluşturuldu")

# Sequence oluştur
X_train_seq = detector.create_sequences(X_train_normal)
print(f"✅ Training sequences: {X_train_seq.shape}")

# Eğit
print("\n🔄 Model eğitimi başlıyor...")
detector.train(
    X_train_seq,
    epochs=50,
    batch_size=32,
    patience=10,
    verbose=1
)

# Model kaydet
model_path = MODELS_DIR / "lstm_autoencoder.h5"
detector.model.save(str(model_path))
print(f"✅ Model kaydedildi: {model_path}")

# === ADIM 5: TEST VE SKORLAMA ===
print("\n" + "="*70)
print("🔍 ADIM 5: TEST VE SKORLAMA")
print("="*70)

# Test sequence oluştur
X_test_seq = detector.create_sequences(X_test_scaled)
print(f"✅ Test sequences: {X_test_seq.shape}")

# Prediction
predictions, scores = detector.predict(X_test_seq)
print(f"✅ Anomali skorları hesaplandı")

# Baseline threshold (median + 3*std)
baseline_threshold = np.median(scores[y_test == 0]) + 3 * np.std(scores[y_test == 0])
print(f"📊 Baseline threshold: {baseline_threshold:.4f}")

# === ADIM 6: THRESHOLD OPTİMİZASYONU ===
print("\n" + "="*70)
print("⚙️ ADIM 6: THRESHOLD OPTİMİZASYONU")
print("="*70)

optimizer = ThresholdOptimizer(y_test, scores)

# Farklı optimizasyon yöntemleri
methods = {
    'Youden': optimizer.optimize_youden(),
    'F1-Score': optimizer.optimize_f1(),
    'Cost-Sensitive': optimizer.optimize_cost_sensitive(
        fp_cost=1, fn_cost=10
    )
}

print("\n📊 Optimizasyon Sonuçları:")
for method_name, result in methods.items():
    print(f"\n{method_name}:")
    print(f"  Threshold: {result['threshold']:.4f}")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall: {result['recall']:.4f}")
    print(f"  F1-Score: {result['f1']:.4f}")

# En iyi threshold'u seç (F1-score'a göre)
best_method = 'F1-Score'
best_threshold = methods[best_method]['threshold']
print(f"\n✅ En iyi threshold ({best_method}): {best_threshold:.4f}")

# Optimize edilmiş prediction
y_pred_optimized = (scores >= best_threshold).astype(int)

# === ADIM 7: DEĞERLENDİRME ===
print("\n" + "="*70)
print("📈 ADIM 7: DEĞERLENDİRME")
print("="*70)

print("\n📊 Classification Report (Optimized Threshold):")
print(classification_report(y_test, y_pred_optimized, 
                           target_names=['Normal', 'Anomaly']))

print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_optimized)
print(cm)

# Metrikler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_optimized),
    'Precision': precision_score(y_test, y_pred_optimized, zero_division=0),
    'Recall': recall_score(y_test, y_pred_optimized, zero_division=0),
    'F1-Score': f1_score(y_test, y_pred_optimized, zero_division=0)
}

print("\n📊 Final Metrics:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")

# === ADIM 8: GÖRSELLEŞTİRME ===
print("\n" + "="*70)
print("📊 ADIM 8: GÖRSELLEŞTİRME")
print("="*70)

# 1. ROC ve PR Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC Curve
axes[0].plot(optimizer.fpr, optimizer.tpr, 'b-', linewidth=2, 
            label=f'ROC (AUC={optimizer.roc_auc:.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# PR Curve
pr_auc = np.trapz(optimizer.precision, optimizer.recall)
axes[1].plot(optimizer.recall, optimizer.precision, 'g-', linewidth=2,
            label=f'PR (AUC={pr_auc:.3f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
roc_pr_path = OUTPUTS_DIR / "roc_pr_curves.png"
plt.savefig(roc_pr_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ ROC/PR curves kaydedildi: {roc_pr_path}")

# 2. Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
cm_path = OUTPUTS_DIR / "confusion_matrix.png"
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Confusion matrix kaydedildi: {cm_path}")

# 3. Score Distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(scores[y_test == 0], bins=50, alpha=0.7, label='Normal', color='green')
plt.hist(scores[y_test == 1], bins=50, alpha=0.7, label='Anomaly', color='red')
plt.axvline(best_threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({best_threshold:.4f})')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.title('Score Distribution', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Training History
if hasattr(detector, 'history') and detector.history is not None:
    plt.subplot(1, 2, 2)
    plt.plot(detector.history.history['loss'], label='Train Loss')
    if 'val_loss' in detector.history.history:
        plt.plot(detector.history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
dist_path = OUTPUTS_DIR / "score_distribution.png"
plt.savefig(dist_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Score distribution kaydedildi: {dist_path}")

# === ADIM 9: RAPOR ===
print("\n" + "="*70)
print("📄 ADIM 9: RAPOR OLUŞTURMA")
print("="*70)

report = {
    'test_date': pd.Timestamp.now().isoformat(),
    'data_info': {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'normal_samples': int((y == 0).sum()),
        'anomaly_samples': int((y == 1).sum()),
        'features': n_features
    },
    'model_info': {
        'model_type': 'LSTM Autoencoder',
        'sequence_length': sequence_length,
        'latent_dim': 16,
        'lstm_units': [64, 32]
    },
    'thresholds': {
        'baseline': float(baseline_threshold),
        'optimized': float(best_threshold),
        'method': best_method
    },
    'metrics': {k: float(v) for k, v in metrics.items()},
    'optimization_results': {
        method: {
            'threshold': float(result['threshold']),
            'precision': float(result['precision']),
            'recall': float(result['recall']),
            'f1': float(result['f1'])
        }
        for method, result in methods.items()
    },
    'confusion_matrix': {
        'tn': int(cm[0, 0]),
        'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0]),
        'tp': int(cm[1, 1])
    }
}

import json
report_path = OUTPUTS_DIR / "test_report.json"
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"✅ Test raporu kaydedildi: {report_path}")

# === ÖZET ===
print("\n" + "="*70)
print("✅ TEST TAMAMLANDI!")
print("="*70)
print(f"\n📊 Sonuçlar:")
print(f"  Accuracy: {metrics['Accuracy']:.4f}")
print(f"  Precision: {metrics['Precision']:.4f}")
print(f"  Recall: {metrics['Recall']:.4f}")
print(f"  F1-Score: {metrics['F1-Score']:.4f}")
print(f"\n📁 Çıktılar:")
print(f"  - Model: {model_path}")
print(f"  - Rapor: {report_path}")
print(f"  - Görseller: {OUTPUTS_DIR}")
print("\n🎉 Pipeline başarıyla test edildi!")
