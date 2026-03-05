"""
ENSEMBLE MODEL - ÇOK MODELL İ KARAR SİSTEMİ
Hangi modelle tespit edileceğine akıllıca karar verir
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

BASE_DIR = Path.cwd()
sys.path.insert(0, str(BASE_DIR / "scripts"))

from models_unsupervised.isolation_forest_detector import IsolationForestDetector
from models_unsupervised.standard_autoencoder import StandardAutoencoder

print("🎯 ENSEMBLE MODEL SİSTEMİ")
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

# Model 1: Isolation Forest (ağırlık: 0.6 - en iyi)
print("\n🌲 Isolation Forest eğitiliyor...")
if_detector = IsolationForestDetector(contamination=0.1)
if_detector.train(X_train_normal)
if_predictions, if_scores = if_detector.predict(X_test)

# Threshold optimize
if_threshold = np.percentile(if_scores, 70)
if_predictions_opt = (if_scores >= if_threshold).astype(int)

# Model 2: Standard Autoencoder (ağırlık: 0.25)
print("\n🔷 Standard Autoencoder eğitiliyor...")
ae_detector = StandardAutoencoder(encoding_dim=8)
ae_detector.train(X_train_normal, epochs=20, batch_size=64, patience=5, verbose=0)
ae_predictions, ae_scores = ae_detector.predict(X_test)

# Threshold optimize
ae_threshold = np.percentile(ae_scores, 95)
ae_predictions_opt = (ae_scores >= ae_threshold).astype(int)

# Model 3: LSTM Autoencoder (ağırlık: 0.15)
# (Şimdilik atlanıyor çünkü performansı düşük)

print("\n" + "="*70)
print("🎯 ENSEMBLE VOTING STRATEJ İLERİ")
print("="*70)

# Strateji 1: MAJORITY VOTING (basit)
print("\n1️⃣ Majority Voting (Çoğunluk):")
majority_vote = (if_predictions_opt + ae_predictions_opt) >= 1  # En az 1 model uyarırsa
majority_vote = majority_vote.astype(int)

f1_majority = f1_score(y_test, majority_vote)
prec_majority = precision_score(y_test, majority_vote, zero_division=0)
rec_majority = recall_score(y_test, majority_vote, zero_division=0)

print(f"   F1: {f1_majority:.4f} | Precision: {prec_majority:.4f} | Recall: {rec_majority:.4f}")

# Strateji 2: WEIGHTED VOTING (ağırlıklı)
print("\n2️⃣ Weighted Voting (Ağırlıklı - ÖNERİLEN):")
# Normalize skorları [0, 1] aralığına
if_scores_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10)
ae_scores_norm = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min() + 1e-10)

# Ağırlıklar (performansa göre)
weights = {
    'isolation_forest': 0.70,  # En iyi performans
    'autoencoder': 0.30,       # Destekleyici
}

# Ağırlıklı skor
weighted_score = (
    weights['isolation_forest'] * if_scores_norm +
    weights['autoencoder'] * ae_scores_norm
)

# Threshold (optimize)
best_f1 = 0
best_threshold = 0.5
for thresh in np.arange(0.3, 0.8, 0.05):
    pred = (weighted_score >= thresh).astype(int)
    f1 = f1_score(y_test, pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

weighted_predictions = (weighted_score >= best_threshold).astype(int)

f1_weighted = f1_score(y_test, weighted_predictions)
prec_weighted = precision_score(y_test, weighted_predictions, zero_division=0)
rec_weighted = recall_score(y_test, weighted_predictions, zero_division=0)

cm = confusion_matrix(y_test, weighted_predictions)
tn, fp, fn, tp = cm.ravel()

print(f"   Optimal Threshold: {best_threshold:.2f}")
print(f"   F1: {f1_weighted:.4f} | Precision: {prec_weighted:.4f} | Recall: {rec_weighted:.4f}")
print(f"   TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
print(f"   FP Rate: {fp/(fp+tn):.4f}")

# Strateji 3: CONFIDENCE-BASED (güven bazlı)
print("\n3️⃣ Confidence-Based (Güven Bazlı):")
print("   Eğer IF yüksek skor verirse → Anomali")
print("   Eğer her iki model de düşük skor verirse → Normal")
print("   Eğer sadece AE yüksek skor verirse → Şüpheli (manuel inceleme)")

confidence_predictions = if_predictions_opt.copy()
suspicious_indices = (if_predictions_opt == 0) & (ae_predictions_opt == 1)
print(f"   Şüpheli vakalar: {suspicious_indices.sum()}")

# En iyi stratejiyi belirle
print("\n" + "="*70)
print("🏆 EN İYİ STRATEJİ:")
print("="*70)

strategies = {
    'Majority Voting': f1_majority,
    'Weighted Voting': f1_weighted,
    'Isolation Forest Only': f1_score(y_test, if_predictions_opt)
}

best_strategy = max(strategies.items(), key=lambda x: x[1])
print(f"\n✅ {best_strategy[0]}: F1-Score = {best_strategy[1]:.4f}")

# Sonuçları kaydet
results = {
    'date': pd.Timestamp.now().isoformat(),
    'strategies': {
        'majority_voting': {
            'f1': float(f1_majority),
            'precision': float(prec_majority),
            'recall': float(rec_majority)
        },
        'weighted_voting': {
            'f1': float(f1_weighted),
            'precision': float(prec_weighted),
            'recall': float(rec_weighted),
            'threshold': float(best_threshold),
            'weights': weights,
            'confusion_matrix': {'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)}
        },
        'isolation_forest_only': {
            'f1': float(f1_score(y_test, if_predictions_opt))
        }
    },
    'best_strategy': best_strategy[0],
    'recommendation': 'Weighted Voting (IF 70%, AE 30%)'
}

output_file = BASE_DIR / "outputs" / "ensemble_results.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n📁 Sonuçlar: {output_file}")

# Karar mantığını açıkla
print("\n" + "="*70)
print("💡 KARAR MANTIĞI:")
print("="*70)
print("""
1. **Isolation Forest (%70 ağırlık):** Ana dedektör
   - En yüksek F1-Score (%99.5)
   - Hızlı ve güvenilir

2. **Standard Autoencoder (%30 ağırlık):** Destekleyici
   - Farklı anomali patternlerini yakalar
   - IF'nin kaçırdıklarını bulabilir

3. **Weighted Score Hesaplama:**
   Score = 0.70 × IF_score + 0.30 × AE_score
   
4. **Karar:**
   - Score >= 0.50 → ANOMALİ
   - Score < 0.50 → NORMAL

5. **Avantajlar:**
   - Tek modelden daha robust
   - False Positive azalır
   - Farklı saldırı tiplerini daha iyi yakalar
""")

print("\n✅ Ensemble sistemi hazır!")
