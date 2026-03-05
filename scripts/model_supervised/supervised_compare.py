"""
🎯 SUPERVISED MODEL COMPARISON (UNSW-NB15 gibi etiketli veri için)
===============================================================

UNSW-NB15'te label var (0/1). En yüksek doğruluk/F1 genelde supervised modellerle gelir.

KULLANIM:
    python scripts/supervised_compare.py
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
DATA_DIR = BASE_DIR / "data"
FEATURES_FILE = DATA_DIR / "features" / "advanced_features.csv"
LABELED_FILE = DATA_DIR / "labeled" / "labeled_traffic.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_xy():
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"Features yok: {FEATURES_FILE} (önce advanced_features.py)")
    if not LABELED_FILE.exists():
        raise FileNotFoundError(f"Labels yok: {LABELED_FILE} (önce label_traffic.py)")

    X = pd.read_csv(FEATURES_FILE)
    df_labels = pd.read_csv(LABELED_FILE)
    y = (df_labels["label"] == "anomaly").astype(int)

    min_len = min(len(X), len(y))
    X = X.iloc[:min_len]
    y = y.iloc[:min_len]

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
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))

    return metrics


if __name__ == "__main__":
    print("=" * 70)
    print("🎯 SUPERVISED MODEL COMPARISON")
    print("=" * 70)

    X, y = load_xy()
    print(f"✅ Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Normal: {(y == 0).sum()} | Anomaly: {(y == 1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
            max_depth=None,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_depth=6,
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

    # Best by F1
    best_name = max(results.keys(), key=lambda k: results[k]["f1"])
    print("\n" + "=" * 70)
    print(f"🏆 BEST MODEL (F1): {best_name} -> {results[best_name]['f1']:.4f}")
    print("=" * 70)

    report = {
        "comparison_date": datetime.now().isoformat(),
        "data_info": {
            "total_samples": int(len(X)),
            "features": int(X.shape[1]),
            "normal_samples": int((y == 0).sum()),
            "anomaly_samples": int((y == 1).sum()),
        },
        "results": results,
        "best_model_by_f1": best_name,
    }

    out = OUTPUTS_DIR / "supervised_comparison_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Report saved: {out}")

