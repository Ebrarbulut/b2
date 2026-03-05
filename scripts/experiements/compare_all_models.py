"""
🔄 TÜM MODELLERİ KARŞILAŞTIRMA SİSTEMİ
======================================

Bu script, tüm anomali tespit modellerini aynı veri setinde test eder
ve performanslarını karşılaştırır.

KULLANIM:
    python scripts/compare_all_models.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Windows console encoding fix (avoid UnicodeEncodeError for emojis / Turkish chars)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Script dizinini path'e ekle
BASE_DIR = Path.cwd()  # Use current working directory
sys.path.insert(0, str(BASE_DIR / "scripts"))

# Modelleri import et (artık models_unsupervised paketinden)
try:
    from models_unsupervised.lstm_autoencoder import LSTMAnomalyDetector
    from models_unsupervised.isolation_forest_detector import IsolationForestDetector
    from models_unsupervised.one_class_svm_detector import OneClassSVMDetector
    from models_unsupervised.standard_autoencoder import StandardAutoencoder
except ImportError as e:
    print(f"❌ Import hatası: {e}")
    print("💡 'scripts/models_unsupervised' altındaki dosyaların mevcut olduğundan emin olun.")
    # Devam et, eksik modelleri atla
    pass

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

print("=" * 70)
print("🔄 TÜM MODELLERİ KARŞILAŞTIRMA")
print("=" * 70)

# === PATHS ===
DATA_DIR = BASE_DIR / "data"
FEATURES_FILE = DATA_DIR / "features" / "advanced_features.csv"
LABELED_FILE = DATA_DIR / "labeled" / "labeled_traffic.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# === VERİ YÜKLEME ===
print("\n" + "="*70)
print("📂 ADIM 1: VERİ YÜKLEME")
print("="*70)

if not FEATURES_FILE.exists():
    print(f"❌ {FEATURES_FILE} bulunamadı!")
    print("💡 Önce 'python scripts/data_pipeline/advanced_features.py' çalıştırın.")
    sys.exit(1)

if not LABELED_FILE.exists():
    print(f"❌ {LABELED_FILE} bulunamadı!")
    print("💡 Önce 'python scripts/data_pipeline/label_traffic.py' çalıştırın.")
    sys.exit(1)

# Feature'ları yükle
X = pd.read_csv(FEATURES_FILE)
print(f"✅ Features loaded: {X.shape}")

# Label'ları yükle
df_labels = pd.read_csv(LABELED_FILE)
y = (df_labels['label'] == 'anomaly').astype(int)
print(f"✅ Labels loaded: {y.shape}")

# Boyutları eşitle
min_len = min(len(X), len(y))
X = X.iloc[:min_len]
y = y.iloc[:min_len]

print(f"✅ Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"   - Normal: {(y == 0).sum()}")
print(f"   - Anomaly: {(y == 1).sum()}")

# === VERİ HAZIRLAMA ===
print("\n" + "="*70)
print("📊 ADIM 2: VERİ HAZIRLAMA")
print("="*70)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train set: {X_train.shape[0]} samples")
print(f"✅ Test set: {X_test.shape[0]} samples")

# Normal trafik (sadece normal ile eğitim için)
X_train_normal = X_train[y_train == 0]
print(f"✅ Normal training samples: {len(X_train_normal)}")

# Normalizasyon (LSTM için gerekli değil ama diğerleri için)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_normal_scaled = X_train_scaled[y_train == 0]

# === MODEL SONUÇLARI ===
results = {}

# === 1. ISOLATION FOREST ===
print("\n" + "="*70)
print("🌲 MODEL 1: ISOLATION FOREST")
print("="*70)

try:
    if_detector = IsolationForestDetector(
        contamination=0.1,
        n_estimators=100,
        random_state=42
    )
    
    if_detector.train(X_train_normal_scaled)
    if_predictions, if_scores = if_detector.predict(X_test_scaled)
    
    # Metrikler
    results['Isolation Forest'] = {
        'accuracy': accuracy_score(y_test, if_predictions),
        'precision': precision_score(y_test, if_predictions, zero_division=0),
        'recall': recall_score(y_test, if_predictions, zero_division=0),
        'f1': f1_score(y_test, if_predictions, zero_division=0),
        'roc_auc': roc_auc_score(y_test, if_scores) if len(np.unique(y_test)) > 1 else 0,
        'predictions': if_predictions,
        'scores': if_scores
    }
    
    print(f"✅ Isolation Forest completed")
    print(f"   F1-Score: {results['Isolation Forest']['f1']:.4f}")
    
except Exception as e:
    print(f"❌ Isolation Forest hatası: {e}")
    results['Isolation Forest'] = None

# === 2. ONE-CLASS SVM ===
print("\n" + "="*70)
print("🔷 MODEL 2: ONE-CLASS SVM")
print("="*70)

try:
    ocsvm_detector = OneClassSVMDetector(
        nu=0.1,
        kernel='rbf',
        gamma='scale'
    )
    
    ocsvm_detector.train(X_train_normal_scaled)
    ocsvm_predictions, ocsvm_scores = ocsvm_detector.predict(X_test_scaled)
    
    # Metrikler
    results['One-Class SVM'] = {
        'accuracy': accuracy_score(y_test, ocsvm_predictions),
        'precision': precision_score(y_test, ocsvm_predictions, zero_division=0),
        'recall': recall_score(y_test, ocsvm_predictions, zero_division=0),
        'f1': f1_score(y_test, ocsvm_predictions, zero_division=0),
        'roc_auc': roc_auc_score(y_test, ocsvm_scores) if len(np.unique(y_test)) > 1 else 0,
        'predictions': ocsvm_predictions,
        'scores': ocsvm_scores
    }
    
    print(f"✅ One-Class SVM completed")
    print(f"   F1-Score: {results['One-Class SVM']['f1']:.4f}")
    
except Exception as e:
    print(f"❌ One-Class SVM hatası: {e}")
    results['One-Class SVM'] = None

# === 3. LSTM AUTOENCODER ===
print("\n" + "="*70)
print("🧠 MODEL 3: LSTM AUTOENCODER")
print("="*70)

try:
    # LSTM için sequence oluştur
    sequence_length = 10
    n_features = X_train_scaled.shape[1]
    
    lstm_detector = LSTMAnomalyDetector(
        sequence_length=sequence_length,
        n_features=n_features,
        latent_dim=16
    )
    
    lstm_detector.build_model(lstm_units=[64, 32])
    
    # Normal trafik ile sequence oluştur
    X_train_normal_seq = lstm_detector.create_sequences(X_train_normal_scaled)
    print(f"   Training sequences: {X_train_normal_seq.shape}")
    
    # Eğit (kısa epoch sayısı - hızlı test için)
    print("   Training (this may take a while)...")
    lstm_detector.train(
        X_train_normal_seq,
        epochs=20,  # Hızlı test için azaltıldı
        batch_size=32,
        patience=5,
        verbose=0
    )
    
    # Test sequence oluştur
    X_test_seq = lstm_detector.create_sequences(X_test_scaled)
    lstm_predictions, lstm_scores = lstm_detector.predict(X_test_seq)
    
    # Metrikler
    results['LSTM Autoencoder'] = {
        'accuracy': accuracy_score(y_test[:len(lstm_predictions)], lstm_predictions),
        'precision': precision_score(y_test[:len(lstm_predictions)], lstm_predictions, zero_division=0),
        'recall': recall_score(y_test[:len(lstm_predictions)], lstm_predictions, zero_division=0),
        'f1': f1_score(y_test[:len(lstm_predictions)], lstm_predictions, zero_division=0),
        'roc_auc': roc_auc_score(y_test[:len(lstm_predictions)], lstm_scores) if len(np.unique(y_test)) > 1 else 0,
        'predictions': lstm_predictions,
        'scores': lstm_scores
    }
    
    print(f"✅ LSTM Autoencoder completed")
    print(f"   F1-Score: {results['LSTM Autoencoder']['f1']:.4f}")
    
except Exception as e:
    print(f"❌ LSTM Autoencoder hatası: {e}")
    import traceback
    traceback.print_exc()
    results['LSTM Autoencoder'] = None

# === 4. STANDARD AUTOENCODER ===
print("\n" + "="*70)
print("🔷 MODEL 4: STANDARD AUTOENCODER")
print("="*70)

try:
    from standard_autoencoder import StandardAutoencoder
    
    std_ae_detector = StandardAutoencoder(
        encoding_dim=16,
        activation='relu'
    )
    
    # Eğit
    print("   Training (this may take a while)...")
    std_ae_detector.train(
        X_train_normal_scaled,
        epochs=30,  # Hızlı test için
        batch_size=32,
        patience=5,
        verbose=0
    )
    
    std_ae_predictions, std_ae_scores = std_ae_detector.predict(X_test_scaled)
    
    # Metrikler
    results['Standard Autoencoder'] = {
        'accuracy': accuracy_score(y_test, std_ae_predictions),
        'precision': precision_score(y_test, std_ae_predictions, zero_division=0),
        'recall': recall_score(y_test, std_ae_predictions, zero_division=0),
        'f1': f1_score(y_test, std_ae_predictions, zero_division=0),
        'roc_auc': roc_auc_score(y_test, std_ae_scores) if len(np.unique(y_test)) > 1 else 0,
        'predictions': std_ae_predictions,
        'scores': std_ae_scores
    }
    
    print(f"✅ Standard Autoencoder completed")
    print(f"   F1-Score: {results['Standard Autoencoder']['f1']:.4f}")
    
except Exception as e:
    print(f"❌ Standard Autoencoder hatası: {e}")
    results['Standard Autoencoder'] = None

# === KARŞILAŞTIRMA TABLOSU ===
print("\n" + "="*70)
print("📊 ADIM 3: KARŞILAŞTIRMA SONUÇLARI")
print("="*70)

# Sonuçları DataFrame'e çevir
comparison_data = []
for model_name, result in results.items():
    if result is not None:
        comparison_data.append({
            'Model': model_name,
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1-Score': result['f1'],
            'ROC-AUC': result['roc_auc']
        })

comparison_df = pd.DataFrame(comparison_data)

if len(comparison_df) > 0:
    print("\n📊 PERFORMANCE COMPARISON:")
    print("=" * 70)
    print(comparison_df.to_string(index=False))
    
    # En iyi model
    best_model = comparison_df.loc[comparison_df['F1-Score'].idxmax()]
    print(f"\n🏆 EN İYİ MODEL (F1-Score): {best_model['Model']}")
    print(f"   F1-Score: {best_model['F1-Score']:.4f}")
    print(f"   Precision: {best_model['Precision']:.4f}")
    print(f"   Recall: {best_model['Recall']:.4f}")
else:
    print("❌ Hiçbir model başarıyla çalıştırılamadı!")

# === GÖRSELLEŞTİRME ===
print("\n" + "="*70)
print("📈 ADIM 4: GÖRSELLEŞTİRME")
print("="*70)

if len(comparison_df) > 0:
    # 1. Metrik karşılaştırması
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        if metric in comparison_df.columns:
            bars = ax.bar(comparison_df['Model'], comparison_df[metric], 
                         color=['#3498db', '#e74c3c', '#2ecc71'][:len(comparison_df)])
            ax.set_ylabel(metric, fontsize=12, fontweight='bold')
            ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            
            # Değerleri bar'ların üzerine yaz
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10)
            
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 6. Confusion Matrix karşılaştırması
    ax = axes[1, 2]
    ax.axis('off')
    ax.text(0.5, 0.5, 'Confusion Matrices\n(see separate plot)', 
           ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    comparison_plot = OUTPUTS_DIR / "model_comparison.png"
    plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Comparison plot saved: {comparison_plot}")
    
    # 2. Confusion Matrix'ler
    n_models = len([r for r in results.values() if r is not None])
    if n_models > 0:
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        idx = 0
        for model_name, result in results.items():
            if result is not None:
                cm = confusion_matrix(y_test[:len(result['predictions'])], 
                                    result['predictions'])
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                           xticklabels=['Normal', 'Anomaly'],
                           yticklabels=['Normal', 'Anomaly'])
                axes[idx].set_title(f'{model_name}\nConfusion Matrix', 
                                   fontsize=12, fontweight='bold')
                axes[idx].set_ylabel('True Label')
                axes[idx].set_xlabel('Predicted Label')
                idx += 1
        
        plt.tight_layout()
        cm_plot = OUTPUTS_DIR / "confusion_matrices.png"
        plt.savefig(cm_plot, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Confusion matrices saved: {cm_plot}")

# === RAPOR KAYDET ===
print("\n" + "="*70)
print("📄 ADIM 5: RAPOR KAYDETME")
print("="*70)

report = {
    'comparison_date': datetime.now().isoformat(),
    'data_info': {
        'total_samples': len(X),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'normal_samples': int((y == 0).sum()),
        'anomaly_samples': int((y == 1).sum()),
        'features': X.shape[1]
    },
    'model_results': {}
}

for model_name, result in results.items():
    if result is not None:
        report['model_results'][model_name] = {
            'accuracy': float(result['accuracy']),
            'precision': float(result['precision']),
            'recall': float(result['recall']),
            'f1_score': float(result['f1']),
            'roc_auc': float(result['roc_auc'])
        }

report_path = OUTPUTS_DIR / "model_comparison_report.json"
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"✅ Report saved: {report_path}")

# CSV olarak da kaydet
if len(comparison_df) > 0:
    csv_path = OUTPUTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"✅ CSV saved: {csv_path}")

# === ÖZET ===
print("\n" + "="*70)
print("✅ KARŞILAŞTIRMA TAMAMLANDI!")
print("="*70)
print(f"\n📊 Test edilen modeller: {len([r for r in results.values() if r is not None])}")
print(f"📁 Çıktılar: {OUTPUTS_DIR}")
print("\n🎉 Tüm modeller başarıyla karşılaştırıldı!")
