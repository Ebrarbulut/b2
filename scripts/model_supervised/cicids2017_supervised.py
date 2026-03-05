"""
🎯 CICIDS2017 - SUPERVISED MODEL COMPARISON
===========================================

Tuesday-WorkingHours.pcap_ISCX.csv dosyası üzerindeki akış (flow) feature'ları
kullanarak LogisticRegression, RandomForest ve HistGradientBoosting modellerini
karşılaştırır.

KULLANIM:
    python scripts/cicids2017_supervised.py
"""

import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Windows console encoding fix
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

import json
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "cicids2017"
CSV_FILE = RAW_DIR / "Tuesday-WorkingHours.pcap_ISCX.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_cicids():
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"CICIDS2017 CSV bulunamadı: {CSV_FILE}")

    print(f"📂 Loading CICIDS2017 CSV: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

    # Sütun adlarını temizle (baş/son boşlukları at, küçük harfe çevir)
    original_cols = list(df.columns)
    df.columns = [c.strip() for c in df.columns]

    # Label kolonu farklı dosyalarda "Label", "label", "class" vb. olabilir
    cols_lower = {c.lower(): c for c in df.columns}
    label_col_name = None
    for cand in ["label", "class", "attack_cat"]:
        if cand in cols_lower:
            label_col_name = cols_lower[cand]
            break

    if label_col_name is None:
        raise ValueError(
            f"CICIDS2017 CSV içinde 'Label' kolonu bulunamadı. Mevcut kolonlar örnek: {original_cols[:10]}"
        )

    # Label'i 0/1'e çevir: BENIGN -> 0, diğer her şey -> 1
    y = (df[label_col_name] != "BENIGN").astype(int)

    # Feature'lar: Label dışındaki tüm kolonlar, numeric'e çevrilip NA=0
    X = df.drop(columns=[label_col_name])
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # NaN -> 0, sonsuz değerleri makul bir aralığa kırp
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)

    # Aşırı uçları sınırlayalım (float64 için güvenli bir sınır)
    X = X.clip(lower=-1e9, upper=1e9)

    return X, y


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    try:
        proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        proba = None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)) if proba is not None and len(np.unique(y_test)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print("\n" + "=" * 70)
    print(f"✅ {name}")
    print("=" * 70)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-Score : {metrics['f1']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["BENIGN", "ATTACK"]))

    return metrics


if __name__ == "__main__":
    print("=" * 70)
    print("🎯 CICIDS2017 - SUPERVISED MODEL COMPARISON")
    print("=" * 70)

    X, y = load_cicids()
    print(f"✅ Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   BENIGN: {(y == 0).sum()} | ATTACK: {(y == 1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),  # sparse güvenli
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
            max_depth=None,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_depth=7,
            random_state=42,
        ),
    }

    results = {}
    for name, model in models.items():
        print("\n" + "-" * 70)
        print(f"Training: {name}")
        print("-" * 70)
        model.fit(X_train, y_train)
        results[name] = evaluate_model(name, model, X_test, y_test)

    best_name = max(results.keys(), key=lambda k: results[k]["f1"])
    print("\n" + "=" * 70)
    print(f"🏆 BEST MODEL (F1): {best_name} -> {results[best_name]['f1']:.4f}")
    print("=" * 70)

    report = {
        "comparison_date": datetime.now().isoformat(),
        "data_info": {
            "total_samples": int(len(X)),
            "features": int(X.shape[1]),
            "benign_samples": int((y == 0).sum()),
            "attack_samples": int((y == 1).sum()),
        },
        "results": results,
        "best_model_by_f1": best_name,
    }

    out = OUTPUTS_DIR / "cicids2017_supervised_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Report saved: {out}")

