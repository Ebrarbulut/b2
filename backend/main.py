"""
PWA / API için FastAPI backend.
Model yüklüyse skor döner, değilse sağlık kontrolü ve demo skor.
"""
import sys
from pathlib import Path

# Proje kökünü path'e ekle (backend klasörünün bir üstü)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Network Anomaly Detection API", version="1.0.0")

# CORS: canlıda frontend adresini ekle (örn. Vercel: https://xxx.vercel.app)
import os
_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = ROOT / "models"

# Model cache (ilk istekte yüklenecek)
_detectors_cache = {}
_feature_names = None


def _load_feature_names():
    """Eğitimde kullanılan feature kolonlarını diskteki advanced_features.csv'den oku."""
    global _feature_names
    if _feature_names is not None:
        return _feature_names
    try:
        import pandas as pd

        features_path = ROOT / "data" / "features" / "advanced_features.csv"
        if features_path.exists():
            df = pd.read_csv(features_path, nrows=1)
            _feature_names = list(df.columns)
    except Exception:
        _feature_names = None
    return _feature_names


def _get_detector(model: Optional[str] = None):
    """
    Kayıtlı modellerden istenen dedektörü (veya ensemble'ı) yükle.

    model:
        - "standard_ae"
        - "isolation_forest"
        - "ensemble"
        - None → varsayılan "ensemble", yoksa "standard_ae", o da yoksa "isolation_forest"
    """
    global _detectors_cache

    from scripts.models_unsupervised.standard_autoencoder import StandardAutoencoder
    from scripts.models_unsupervised.isolation_forest_detector import IsolationForestDetector
    from scripts.monitoring.ensemble_detector import EnsembleDetector

    # Hangi model isteniyor?
    requested = (model or "ensemble").lower()

    # Cache'te varsa direkt dön
    if requested in _detectors_cache:
        return _detectors_cache[requested]

    feature_names = _load_feature_names()

    # Tekil modelleri lazy-load et
    if "standard_ae" not in _detectors_cache:
        try:
            ae = StandardAutoencoder.load_model(
                model_path=str(MODELS_DIR / "standard_autoencoder.keras"),
                scaler_path=str(MODELS_DIR / "standard_ae_scaler.pkl"),
                config_path=str(ROOT / "standard_ae_config.pkl"),
            )
            _detectors_cache["standard_ae"] = (ae, feature_names, "standard_ae")
        except Exception:
            _detectors_cache["standard_ae"] = (None, feature_names, "standard_ae")

    if "isolation_forest" not in _detectors_cache:
        try:
            if_detector = IsolationForestDetector.load_model(
                model_path=str(MODELS_DIR / "isolation_forest.pkl"),
                scaler_path=str(MODELS_DIR / "if_scaler.pkl"),
                config_path=str(ROOT / "if_config.pkl"),
            )
            _detectors_cache["isolation_forest"] = (if_detector, feature_names, "isolation_forest")
        except Exception:
            _detectors_cache["isolation_forest"] = (None, feature_names, "isolation_forest")

    # Ensemble: mevcut olan modellerden birini bile alsa çalışsın
    if "ensemble" not in _detectors_cache:
        ae, _, _ = _detectors_cache.get("standard_ae", (None, None, None))
        if_detector, _, _ = _detectors_cache.get("isolation_forest", (None, None, None))
        ensemble = None
        if ae is not None or if_detector is not None:
            try:
                ensemble = EnsembleDetector()
                if if_detector is not None:
                    ensemble.add_model("isolation_forest", if_detector, weight=1.0)
                if ae is not None:
                    ensemble.add_model("standard_ae", ae, weight=1.0)
                _detectors_cache["ensemble"] = (ensemble, feature_names, "ensemble")
            except Exception:
                _detectors_cache["ensemble"] = (None, feature_names, "ensemble")
        else:
            _detectors_cache["ensemble"] = (None, feature_names, "ensemble")

    # İstenen model yoksa fallback sırası
    for key in [requested, "ensemble", "standard_ae", "isolation_forest"]:
        det_tuple = _detectors_cache.get(key)
        if det_tuple and det_tuple[0] is not None:
            return det_tuple

    # Hiçbiri yoksa demo moda düş
    return None, feature_names, "demo"


class ScoreRequest(BaseModel):
    """İstek: feature vektörleri (her biri 12 eleman)."""
    features: List[List[float]]
    model: Optional[str] = None  # "standard_ae", "isolation_forest", "ensemble"


class ScoreResponse(BaseModel):
    """Yanıt: her örnek için anomali skoru."""
    scores: List[float]
    model: str


@app.get("/health")
def health():
    """API çalışıyor mu kontrolü."""
    return {"status": "ok", "service": "nad-api"}


@app.get("/api/models")
def list_models():
    """Sunucuda hangi modellerin dosyalarının hazır olduğunu döner (ağır yükleme yapmadan)."""
    models = []
    if (MODELS_DIR / "standard_autoencoder.keras").exists() and (
        MODELS_DIR / "standard_ae_scaler.pkl"
    ).exists() and (ROOT / "standard_ae_config.pkl").exists():
        models.append("standard_ae")
    if (
        (MODELS_DIR / "isolation_forest.pkl").exists()
        and (MODELS_DIR / "if_scaler.pkl").exists()
        and (ROOT / "if_config.pkl").exists()
    ):
        models.append("isolation_forest")
    ensemble_available = len(models) >= 1
    return {
        "models": models,
        "ensemble_available": ensemble_available,
    }


@app.post("/api/score", response_model=ScoreResponse)
def score(request: ScoreRequest):
    """Feature matrisi alır, anomali skorları döner."""
    detector, feature_names, model_name = _get_detector(request.model)
    if not request.features:
        raise HTTPException(status_code=400, detail="features boş olamaz")
    # Numpy/pandas/sklearn bağımlılıklarını zorunlu kılmamak için
    # istek verisini saf Python listeleriyle işleriz.
    try:
        X = [[float(v) for v in row] for row in request.features]
    except Exception:
        raise HTTPException(status_code=400, detail="features sayısal değerlerden oluşmalı")

    n_rows = len(X)
    n_cols = len(X[0]) if n_rows > 0 else 0
    if any(len(row) != n_cols for row in X):
        raise HTTPException(status_code=400, detail="features satır uzunlukları aynı olmalı")

    if detector is not None and feature_names is not None and model_name != "demo":
        if n_cols != len(feature_names):
            raise HTTPException(
                status_code=400,
                detail=f"Feature sayısı {len(feature_names)} olmalı: {n_cols}"
            )
        try:
            # EnsembleDetector veya tekil model: predict(X) -> (predictions, scores)
            _, scores = detector.predict(X)
            if hasattr(scores, "tolist"):
                scores_out = scores.tolist()
            else:
                scores_out = [float(s) for s in scores]
            return ScoreResponse(scores=scores_out, model=model_name)
        except Exception:
            # Model pipeline'ı ortamda çalışmıyorsa demo moda düş.
            detector = None
            model_name = "demo"
    # Model yoksa demo: rastgele benzeri skorlar
    import random
    scores = [random.uniform(0.0, 0.5) for _ in range(len(request.features))]
    return ScoreResponse(scores=scores, model=model_name)
