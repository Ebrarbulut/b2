"""
FEATURE IMPORTANCE ANALİZİ
Hangi özelliklerin anomali tespitinde önemli olduğunu göster
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

BASE_DIR = Path.cwd()

print("🔍 FEATURE IMPORTANCE ANALİZİ")
print("="*70)

# Veri yükle
X = pd.read_csv(BASE_DIR / "data/features/advanced_features.csv")
df_labels = pd.read_csv(BASE_DIR / "data/labeled/labeled_traffic.csv")
y = (df_labels['label'] == 'anomaly').astype(int).values

feature_names = X.columns.tolist()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# RandomForest ile importance hesapla
print("\n🌲 RandomForest ile feature importance hesaplanıyor...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Sonuçları yazdır
print("\n📊 FEATURE IMPORTANCE SIRALAMAS I:")
print("="*70)

importance_dict = {}
for i, idx in enumerate(indices, 1):
    importance = importances[idx]
    importance_dict[feature_names[idx]] = float(importance)
    print(f"{i:2d}. {feature_names[idx]:25s} : {importance:.4f}")

# Görselleştir
plt.figure(figsize=(12, 8))
plt.barh(range(len(indices)), importances[indices], color='skyblue', edgecolor='black')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Importance Score', fontsize=12)
plt.title('Feature Importance for Anomaly Detection', fontsize=14, fontweight='bold')
plt.tight_layout()

output_dir = BASE_DIR / "outputs"
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
print(f"\n✅ Grafik kaydedildi: {output_dir / 'feature_importance.png'}")

# JSON olarak kaydet
results = {
    'date': pd.Timestamp.now().isoformat(),
    'method': 'RandomForest',
    'feature_importance': importance_dict,
    'top_5_features': [feature_names[idx] for idx in indices[:5]],
    'model_accuracy': float(rf.score(X_test, y_test))
}

with open(output_dir / "feature_importance.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Sonuçlar kaydedildi: {output_dir / 'feature_importance.json'}")

# En önemli 5 özellik
print(f"\n🏆 EN ÖNEMLİ 5 ÖZELLİK:")
for i, idx in enumerate(indices[:5], 1):
    print(f"   {i}. {feature_names[idx]} ({importances[idx]:.4f})")
