from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.models_unsupervised.standard_autoencoder import StandardAutoencoder
from scripts.models_unsupervised.isolation_forest_detector import IsolationForestDetector
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"


def load_dataset():
    features_path = DATA_DIR / "features" / "advanced_features.csv"
    labels_path = DATA_DIR / "labeled" / "labeled_traffic.csv"

    if not features_path.exists():
        raise FileNotFoundError(f"features dosyası bulunamadı: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"label dosyası bulunamadı: {labels_path}")

    X = pd.read_csv(features_path)
    df_labels = pd.read_csv(labels_path)

    # Label'i 0/1'e çevir (0: normal, 1: anomaly)
    y = (df_labels["label"] == "anomaly").astype(int).values

    # Güvenlik: uzunlukları hizala
    n = min(len(X), len(y))
    X = X.iloc[:n].reset_index(drop=True)
    y = y[:n]

    return X, y


def evaluate_binary(y_true, scores, threshold=None):
    if threshold is None:
        # Skor daha yüksek = daha riskli varsayımıyla, üst %5'i anomaly say
        thr = np.percentile(scores, 95)
    else:
        thr = threshold

    y_pred = (scores > thr).astype(int)

    metrics = {
        "threshold": float(thr),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # ROC-AUC için hem 0 hem 1 sınıfı olmalı
    if len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, scores)
        except Exception:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")

    # Yanlış pozitif oranı
    normal_mask = y_true == 0
    if normal_mask.any():
        fp = np.logical_and(y_pred == 1, y_true == 0).sum()
        metrics["false_positive_rate"] = fp / normal_mask.sum()
    else:
        metrics["false_positive_rate"] = float("nan")

    return metrics


def run_comparison():
    print("Veri yükleniyor...")
    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_normal = X_train[y_train == 0]

    results: list[dict] = []

    # === Standard Autoencoder ===
    print("\nStandard Autoencoder değerlendiriliyor...")
    ae = StandardAutoencoder(encoding_dim=16)
    ae.build_model(input_dim=X_train_normal.shape[1])
    ae.train(
        X_train_normal.values,
        epochs=20,
        batch_size=64,
        patience=5,
        verbose=0,
    )

    ae_scores = ae.predict_anomaly_scores(X_test.values)
    ae_metrics = evaluate_binary(y_test, ae_scores, threshold=ae.threshold)
    ae_metrics["model"] = "standard_autoencoder"
    results.append(ae_metrics)

    # Model dosyalarını kaydet (backend ve realtime NIDS için)
    MODELS_DIR.mkdir(exist_ok=True)
    ae.save_model(
        model_path=str(MODELS_DIR / "standard_autoencoder.keras"),
        scaler_path=str(MODELS_DIR / "standard_ae_scaler.pkl"),
    )

    # === Isolation Forest ===
    print("\nIsolation Forest değerlendiriliyor...")
    if_detector = IsolationForestDetector(contamination=0.05)
    # Çok büyük veri setlerinde hız için upper bound koy
    max_train = min(20000, len(X_train_normal))
    if_detector.train(X_train_normal.iloc[:max_train].values)

    if_scores = if_detector.predict_anomaly_scores(X_test.values)
    if_metrics = evaluate_binary(y_test, if_scores, threshold=if_detector.threshold)
    if_metrics["model"] = "isolation_forest"
    results.append(if_metrics)

    if_detector.save_model(
        model_path=str(MODELS_DIR / "isolation_forest.pkl"),
        scaler_path=str(MODELS_DIR / "if_scaler.pkl"),
    )

    # === Sonuçları tabloya yaz ===
    OUTPUTS_DIR.mkdir(exist_ok=True)

    df_results = pd.DataFrame(results).set_index("model")
    out_path = OUTPUTS_DIR / "core_model_comparison.csv"
    df_results.to_csv(out_path)

    print("\nÇekirdek modeller karşılaştırması:")
    print(df_results.round(4))
    print(f"\nSonuçlar kaydedildi: {out_path}")


if __name__ == "__main__":
    run_comparison()

