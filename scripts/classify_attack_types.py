"""
SALDIRI TİPİ SINIFLANDIRMASI
Anomali tespit edildikten sonra hangi saldırı tipi olduğunu belirle
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import json

BASE_DIR = Path.cwd()

print("🎯 SALDIRI TİPİ SINIFLANDIRMASI")
print("="*70)

# Veri yükle
X = pd.read_csv(BASE_DIR / "data/features/advanced_features.csv")
df_labels = pd.read_csv(BASE_DIR / "data/labeled/labeled_traffic.csv")

# Sadece anomali örneklerini al
anomaly_mask = df_labels['label'] == 'anomaly'
X_anomaly = X[anomaly_mask]
attack_types = df_labels[anomaly_mask]['attack_type'].values

print(f"📊 Toplam anomali: {len(X_anomaly)}")
print(f"   Saldırı tipleri: {np.unique(attack_types)}")

# Attack type dağılımı
print("\n📈 Saldırı Tipi Dağılımı:")
for attack_type in np.unique(attack_types):
    count = (attack_types == attack_type).sum()
    print(f"   {attack_type}: {count} ({count/len(attack_types)*100:.1f}%)")

# Label encode
le = LabelEncoder()
y_encoded = le.fit_transform(attack_types)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_anomaly, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n✅ Train: {len(y_train)} | Test: {len(y_test)}")

# RandomForest sınıflandırıcı
print("\n🌲 RandomForest ile saldırı tipi sınıflandırması eğitiliyor...")
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# Tahmin
y_pred = clf.predict(X_test)

# Değerlendirme
print("\n📊 SINIFLANDIRMA RAPORU:")
print("="*70)
print(classification_report(
    y_test, y_pred,
    target_names=le.classes_,
    digits=4
))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n📊 Confusion Matrix:")
print(cm)

# Accuracy
accuracy = (y_test == y_pred).sum() / len(y_test)
print(f"\n🎯 Genel Doğruluk: {accuracy:.4f}")

# Sonuçları kaydet
results = {
    'date': pd.Timestamp.now().isoformat(),
    'model': 'RandomForest',
    'attack_types': le.classes_.tolist(),
    'accuracy': float(accuracy),
    'n_train': len(y_train),
    'n_test': len(y_test),
    'confusion_matrix': cm.tolist()
}

output_file = BASE_DIR / "outputs" / "attack_classification.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Sonuçlar kaydedildi: {output_file}")

# Model kaydet
import pickle
model_path = BASE_DIR / "models" / "attack_classifier.pkl"
model_path.parent.mkdir(exist_ok=True)

with open(model_path, 'wb') as f:
    pickle.dump({'model': clf, 'label_encoder': le}, f)

print(f"✅ Model kaydedildi: {model_path}")
