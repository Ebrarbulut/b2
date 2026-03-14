"""
🎯 CICIDS2017 CROSS-DAY TEST
============================

Amaç: Overfitting / leakage olmasın diye:
  - Train: Tuesday-WorkingHours.pcap_ISCX.csv
  - Test : Wednesday-workingHours.pcap_ISCX.csv

KULLANIM:
    python scripts/cicids2017_cross_day.py
"""

import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

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
TUESDAY = RAW_DIR / "Tuesday-WorkingHours.pcap_ISCX.csv"
WEDNESDAY = RAW_DIR / "Wednesday-workingHours.pcap_ISCX.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_day(csv_path: Path):
    print(f"📂 Loading {csv_path.name}")
    df = pd.read_csv(csv_path)
    print(f"   rows={len(df)}, cols={len(df.columns)}")

    # Sütun isimlerini temizle
    original_cols = list(df.columns)
    df.columns = [c.strip() for c in df.columns]
    cols_lower = {c.lower(): c for c in df.columns}

    label_col = None
    for cand in ["label", "class", "attack_cat"]:
        if cand in cols_lower:
            label_col = cols_lower[cand]
            break
    if label_col is None:
        raise ValueError(f"{csv_path.name} içinde label kolonu bulunamadı. Örnek kolonlar: {original_cols[:10]}")

    y = (df[label_col] != "BENIGN").astype(int)

    X = df.drop(columns=[label_col])
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X.clip(lower=-1e9, upper=1e9)

    return X, y


def evaluate(name, model, X_test, y_test):
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
    print(f"✅ {name} (Train: Tuesday, Test: Wednesday)")
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
    print("🎯 CICIDS2017 CROSS-DAY TEST (Tuesday -> Wednesday)")
    print("=" * 70)

    if not TUESDAY.exists() or not WEDNESDAY.exists():
        raise FileNotFoundError("Tuesday veya Wednesday CSV bulunamadı (data/raw/cicids2017 altında).")

    X_train, y_train = load_day(TUESDAY)
    X_test, y_test = load_day(WEDNESDAY)

    print(f"\n✅ Train: {len(X_train)} samples (Tuesday) | BENIGN={(y_train==0).sum()} ATTACK={(y_train==1).sum()}")
    print(f"✅ Test : {len(X_test)} samples (Wednesday) | BENIGN={(y_test==0).sum()} ATTACK={(y_test==1).sum()}")

    models = {
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
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
        print(f"Training: {name} (Tuesday)")
        print("-" * 70)
        model.fit(X_train, y_train)
        results[name] = evaluate(name, model, X_test, y_test)

    best = max(results.keys(), key=lambda k: results[k]["f1"])
    print("\n" + "=" * 70)
    print(f"🏆 BEST MODEL (F1) - CROSS-DAY: {best} -> {results[best]['f1']:.4f}")
    print("=" * 70)

    report = {
        "comparison_date": datetime.now().isoformat(),
        "train_file": TUESDAY.name,
        "test_file": WEDNESDAY.name,
        "train_info": {
            "total_samples": int(len(X_train)),
            "benign_samples": int((y_train == 0).sum()),
            "attack_samples": int((y_train == 1).sum()),
        },
        "test_info": {
            "total_samples": int(len(X_test)),
            "benign_samples": int((y_test == 0).sum()),
            "attack_samples": int((y_test == 1).sum()),
        },
        "results": results,
        "best_model_by_f1": best,
    }

    out = OUTPUTS_DIR / "cicids2017_cross_day_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Report saved: {out}")

