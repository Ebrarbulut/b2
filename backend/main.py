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

# Model global (ilk istekte yüklenecek)
_detector = None
_feature_names = None


def _get_detector():
    """Varsa Isolation Forest veya basit skorlayıcı yükle."""
    global _detector, _feature_names
    if _detector is not None:
        return _detector, _feature_names
    try:
        from scripts.models_unsupervised.isolation_forest_detector import IsolationForestDetector
        import pandas as pd
        features_path = ROOT / "data" / "features" / "advanced_features.csv"
        if features_path.exists():
            X = pd.read_csv(features_path)
            if len(X) > 100:
                _feature_names = list(X.columns)
                _detector = IsolationForestDetector(contamination=0.1)
                _detector.train(X.head(5000))
                return _detector, _feature_names
    except Exception:
        pass
    return None, None


class ScoreRequest(BaseModel):
    """İstek: feature vektörleri (her biri 12 eleman)."""
    features: List[List[float]]


class ScoreResponse(BaseModel):
    """Yanıt: her örnek için anomali skoru."""
    scores: List[float]
    model: str


@app.get("/health")
def health():
    """API çalışıyor mu kontrolü."""
    return {"status": "ok", "service": "nad-api"}


@app.post("/api/score", response_model=ScoreResponse)
def score(request: ScoreRequest):
    """Feature matrisi alır, anomali skorları döner."""
    detector, feature_names = _get_detector()
    if not request.features:
        raise HTTPException(status_code=400, detail="features boş olamaz")
    import numpy as np
    X = np.array(request.features, dtype=float)
    if detector is not None and feature_names is not None:
        if X.shape[1] != len(feature_names):
            raise HTTPException(
                status_code=400,
                detail=f"Feature sayısı {X.shape[1]} olmalı: {len(feature_names)}"
            )
        _, scores = detector.predict(X)
        return ScoreResponse(scores=scores.tolist(), model="IsolationForest")
    # Model yoksa demo: rastgele benzeri skorlar
    import random
    scores = [random.uniform(0.0, 0.5) for _ in range(len(request.features))]
    return ScoreResponse(scores=scores, model="demo")
