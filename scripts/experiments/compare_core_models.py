"""
Tum modelleri egitir, karsilastirir ve models/ + kokte config dosyalarini uretir.
Modeller: Standard AE, Isolation Forest, One-Class SVM, LSTM AE (seq=1), Ensemble.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"


def load_dataset():
    features_path = DATA_DIR / "features" / "advanced_features.csv"
    labels_path = DATA_DIR / "labeled" / "labeled_traffic.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"features dosyasi bulunamadi: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"label dosyasi bulunamadi: {labels_path}")
    X = pd.read_csv(features_path)
    df_labels = pd.read_csv(labels_path)
    y = (df_labels["label"] == "anomaly").astype(int).values
    n = min(len(X), len(y))
    X = X.iloc[:n].reset_index(drop=True)
    y = y[:n]
    return X, y


def evaluate_binary(y_true, scores, threshold=None):
    if threshold is None:
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
    if len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, scores)
        except Exception:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
    normal_mask = y_true == 0
    if normal_mask.any():
        fp = np.logical_and(y_pred == 1, y_true == 0).sum()
        metrics["false_positive_rate"] = fp / normal_mask.sum()
    else:
        metrics["false_positive_rate"] = float("nan")
    return metrics


def run_comparison():
    import os
    os.chdir(BASE_DIR)
    print("Veri yukleniyor...")
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_normal = X_train[y_train == 0]
    n_features = X_train_normal.shape[1]
    MODELS_DIR.mkdir(exist_ok=True)
    results = []

    # --- Standard Autoencoder ---
    print("\n[1/4] Standard Autoencoder...")
    from scripts.models_unsupervised.standard_autoencoder import StandardAutoencoder
    ae = StandardAutoencoder(encoding_dim=16)
    ae.build_model(input_dim=n_features)
    ae.train(X_train_normal.values, epochs=20, batch_size=64, patience=5, verbose=0)
    ae_scores = ae.predict_anomaly_scores(X_test.values)
    results.append({**evaluate_binary(y_test, ae_scores, ae.threshold), "model": "standard_ae"})
    ae.save_model(
        model_path=str(MODELS_DIR / "standard_autoencoder.keras"),
        scaler_path=str(MODELS_DIR / "standard_ae_scaler.pkl"),
    )

    # --- Isolation Forest ---
    print("\n[2/4] Isolation Forest...")
    from scripts.models_unsupervised.isolation_forest_detector import IsolationForestDetector
    if_det = IsolationForestDetector(contamination=0.05)
    max_tr = min(20000, len(X_train_normal))
    if_det.train(X_train_normal.iloc[:max_tr].values)
    if_scores = if_det.predict_anomaly_scores(X_test.values)
    results.append({**evaluate_binary(y_test, if_scores, if_det.threshold), "model": "isolation_forest"})
    if_det.save_model(
        model_path=str(MODELS_DIR / "isolation_forest.pkl"),
        scaler_path=str(MODELS_DIR / "if_scaler.pkl"),
    )

    # --- One-Class SVM ---
    print("\n[3/4] One-Class SVM...")
    from scripts.models_unsupervised.one_class_svm_detector import OneClassSVMDetector
    ocsvm = OneClassSVMDetector(nu=0.05, kernel="rbf")
    ocsvm.train(X_train_normal.iloc[:max_tr].values)
    oc_scores = ocsvm.predict_anomaly_scores(X_test.values)
    results.append({**evaluate_binary(y_test, oc_scores, ocsvm.threshold), "model": "one_class_svm"})
    ocsvm.save_model(
        model_path=str(MODELS_DIR / "one_class_svm.pkl"),
        scaler_path=str(MODELS_DIR / "ocsvm_scaler.pkl"),
    )

    # --- LSTM AE (sequence_length=1) ---
    print("\n[4/4] LSTM Autoencoder (seq=1)...")
    from scripts.models_unsupervised.lstm_autoencoder import LSTMAnomalyDetector
    seq_len = 1
    X_train_3d = X_train_normal.values.reshape(-1, seq_len, n_features)
    X_test_3d = X_test.values.reshape(-1, seq_len, n_features)
    lstm = LSTMAnomalyDetector(sequence_length=seq_len, n_features=n_features, latent_dim=16)
    lstm.build_model(lstm_units=[32, 16])
    lstm.train(X_train_3d, epochs=15, batch_size=64, patience=4, verbose=0)
    lstm_scores = lstm.predict_anomaly_scores(X_test_3d)
    results.append({**evaluate_binary(y_test, lstm_scores, lstm.threshold), "model": "lstm_ae"})
    lstm.save_model(
        model_path=str(MODELS_DIR / "lstm_autoencoder.keras"),
        scaler_path=str(MODELS_DIR / "lstm_scaler.pkl"),
    )

    # --- Sonuc ---
    OUTPUTS_DIR.mkdir(exist_ok=True)
    df_results = pd.DataFrame(results).set_index("model")
    out_path = OUTPUTS_DIR / "core_model_comparison.csv"
    df_results.to_csv(out_path)
    print("\nTum modeller karsilastirma:")
    print(df_results.round(4))
    print(f"\nSonuclar: {out_path}")


if __name__ == "__main__":
    run_comparison()
