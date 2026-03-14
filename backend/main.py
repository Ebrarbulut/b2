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

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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
OUTPUTS_DIR = ROOT / "outputs"

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


def _flow_key(pkt):
    from scapy.layers.inet import IP, TCP, UDP, ICMP  # type: ignore

    if IP not in pkt:
        return None
    ip = pkt[IP]
    proto = "other"
    sport = 0
    dport = 0
    if TCP in pkt:
        proto = "tcp"
        sport = int(pkt[TCP].sport)
        dport = int(pkt[TCP].dport)
    elif UDP in pkt:
        proto = "udp"
        sport = int(pkt[UDP].sport)
        dport = int(pkt[UDP].dport)
    elif ICMP in pkt:
        proto = "icmp"
    return (ip.src, sport, ip.dst, dport, proto)


def _pcap_to_feature_rows(raw_bytes: bytes, feature_names: Optional[List[str]]):
    """
    PCAP içindeki paketleri basit flow istatistiklerine çevirir.
    realtime_nids_scapy.py içindeki mantığın sadeleştirilmiş versiyonu.
    """
    if feature_names is None:
        return []

    from scapy.utils import PcapReader  # type: ignore
    import numpy as np
    import pandas as pd
    import tempfile

    # Scapy, bellekteki bytes ile doğrudan stabil çalışmadığı için geçici dosya kullanıyoruz.
    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=True) as tmp:
        tmp.write(raw_bytes)
        tmp.flush()
        flows = {}
        with PcapReader(tmp.name) as pcap:
            for pkt in pcap:
                key = _flow_key(pkt)
                if key is None:
                    continue
                now = float(getattr(pkt, "time", 0.0))
                length = int(len(pkt))
                if key not in flows:
                    src_ip, src_p, dst_ip, dst_p, proto = key
                    flows[key] = {
                        "ts_start": now,
                        "ts_end": now,
                        "id.orig_h": src_ip,
                        "id.orig_p": src_p,
                        "id.resp_h": dst_ip,
                        "id.resp_p": dst_p,
                        "proto": proto,
                        "orig_bytes": length,
                        "resp_bytes": 0.0,
                        "orig_pkts": 1.0,
                        "resp_pkts": 0.0,
                    }
                else:
                    f = flows[key]
                    f["ts_end"] = now
                    f["orig_bytes"] += length
                    f["orig_pkts"] += 1.0

    if not flows:
        return []

    rows = []
    for _, f in flows.items():
        duration = max(f["ts_end"] - f["ts_start"], 0.0)
        orig_bytes = float(f["orig_bytes"])
        resp_bytes = float(f["resp_bytes"])
        orig_pkts = float(f["orig_pkts"])
        resp_pkts = float(f["resp_pkts"])
        total_bytes = orig_bytes + resp_bytes
        total_pkts = orig_pkts + resp_pkts
        bytes_ratio = orig_bytes / resp_bytes if resp_bytes > 0 else 0.0
        pkts_ratio = orig_pkts / resp_pkts if resp_pkts > 0 else 0.0

        base_row = {
            "duration": duration,
            "orig_bytes": orig_bytes,
            "resp_bytes": resp_bytes,
            "orig_pkts": orig_pkts,
            "resp_pkts": resp_pkts,
            "total_bytes": total_bytes,
            "total_pkts": total_pkts,
            "bytes_ratio": bytes_ratio,
            "pkts_ratio": pkts_ratio,
            "src_port_entropy": 0.0,
            "dst_port_entropy": 0.0,
            "connections_per_min": 0.0,
        }
        ordered = {col: base_row.get(col, 0.0) for col in feature_names}
        rows.append(ordered)

    df = pd.DataFrame(rows)
    return df.values.astype(np.float32)


def _get_detector(model: Optional[str] = None):
    """
    Kayıtlı modellerden istenen dedektörü (veya ensemble'ı) yükle.
    Desteklenen: standard_ae, isolation_forest, one_class_svm, lstm_ae, ensemble.
    """
    global _detectors_cache

    from scripts.models_unsupervised.standard_autoencoder import StandardAutoencoder
    from scripts.models_unsupervised.isolation_forest_detector import IsolationForestDetector
    from scripts.models_unsupervised.one_class_svm_detector import OneClassSVMDetector
    from scripts.models_unsupervised.lstm_autoencoder import LSTMAnomalyDetector
    from scripts.monitoring.ensemble_detector import EnsembleDetector

    requested = (model or "ensemble").lower()
    if requested in _detectors_cache:
        return _detectors_cache[requested]

    feature_names = _load_feature_names()

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
            if_det = IsolationForestDetector.load_model(
                model_path=str(MODELS_DIR / "isolation_forest.pkl"),
                scaler_path=str(MODELS_DIR / "if_scaler.pkl"),
                config_path=str(ROOT / "if_config.pkl"),
            )
            _detectors_cache["isolation_forest"] = (if_det, feature_names, "isolation_forest")
        except Exception:
            _detectors_cache["isolation_forest"] = (None, feature_names, "isolation_forest")

    if "one_class_svm" not in _detectors_cache:
        try:
            ocsvm = OneClassSVMDetector.load_model(
                model_path=str(MODELS_DIR / "one_class_svm.pkl"),
                scaler_path=str(MODELS_DIR / "ocsvm_scaler.pkl"),
                config_path=str(ROOT / "ocsvm_config.pkl"),
            )
            _detectors_cache["one_class_svm"] = (ocsvm, feature_names, "one_class_svm")
        except Exception:
            _detectors_cache["one_class_svm"] = (None, feature_names, "one_class_svm")

    if "lstm_ae" not in _detectors_cache:
        try:
            lstm = LSTMAnomalyDetector.load_model(
                model_path=str(MODELS_DIR / "lstm_autoencoder.keras"),
                scaler_path=str(MODELS_DIR / "lstm_scaler.pkl"),
                config_path=str(ROOT / "lstm_config.pkl"),
            )
            _detectors_cache["lstm_ae"] = (lstm, feature_names, "lstm_ae")
        except Exception:
            _detectors_cache["lstm_ae"] = (None, feature_names, "lstm_ae")

    if "ensemble" not in _detectors_cache:
        ensemble = None
        try:
            ens = EnsembleDetector()
            for name in ["isolation_forest", "standard_ae", "one_class_svm", "lstm_ae"]:
                det, _, _ = _detectors_cache.get(name, (None, None, None))
                if det is not None:
                    ens.add_model(name, det, weight=1.0)
            if len(ens.models) > 0:
                ensemble = ens
        except Exception:
            pass
        _detectors_cache["ensemble"] = (ensemble, feature_names, "ensemble")

    for key in [requested, "ensemble", "standard_ae", "isolation_forest", "one_class_svm", "lstm_ae"]:
        det_tuple = _detectors_cache.get(key)
        if det_tuple and det_tuple[0] is not None:
            return det_tuple
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


@app.get("/api/comparison")
def get_comparison():
    """Lokalde compare_core_models.py çalıştırıldıysa outputs/core_model_comparison.csv'den metrikleri döner."""
    import csv
    path = OUTPUTS_DIR / "core_model_comparison.csv"
    if not path.exists():
        return {"available": False, "rows": []}
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out = {}
            for k, v in row.items():
                if k == "model" or v is None or v == "":
                    out[k] = v
                elif str(v).lower() == "nan":
                    out[k] = None
                else:
                    try:
                        out[k] = float(v)
                    except ValueError:
                        out[k] = v
            rows.append(out)
    return {"available": True, "rows": rows}


@app.get("/api/models")
def list_models():
    """Sunucuda hangi modellerin dosyalarının hazır olduğunu döner."""
    models = []
    if (MODELS_DIR / "standard_autoencoder.keras").exists() and (MODELS_DIR / "standard_ae_scaler.pkl").exists() and (ROOT / "standard_ae_config.pkl").exists():
        models.append("standard_ae")
    if (MODELS_DIR / "isolation_forest.pkl").exists() and (MODELS_DIR / "if_scaler.pkl").exists() and (ROOT / "if_config.pkl").exists():
        models.append("isolation_forest")
    if (MODELS_DIR / "one_class_svm.pkl").exists() and (MODELS_DIR / "ocsvm_scaler.pkl").exists() and (ROOT / "ocsvm_config.pkl").exists():
        models.append("one_class_svm")
    if (MODELS_DIR / "lstm_autoencoder.keras").exists() and (MODELS_DIR / "lstm_scaler.pkl").exists() and (ROOT / "lstm_config.pkl").exists():
        models.append("lstm_ae")
    return {"models": models, "ensemble_available": len(models) >= 1}


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
            import numpy as np
            X_arr = np.array(X, dtype=np.float32)
            if model_name == "lstm_ae":
                X_arr = X_arr.reshape(-1, 1, n_cols)
            _, scores = detector.predict(X_arr)
            if hasattr(scores, "tolist"):
                scores_out = scores.tolist()
            else:
                scores_out = [float(s) for s in scores]
            return ScoreResponse(scores=scores_out, model=model_name)
        except Exception:
            detector = None
            model_name = "demo"
    # Model yoksa demo: rastgele benzeri skorlar
    import random
    scores = [random.uniform(0.0, 0.5) for _ in range(len(request.features))]
    return ScoreResponse(scores=scores, model=model_name)


@app.post("/api/analyze-csv")
async def analyze_csv(file: UploadFile = File(...), model: Optional[str] = Form(None)):
  """
  Kullanıcının yüklediği CSV dosyasını seçilen modelle toplu analiz eder.
  Beklenti: kolonlar eğitimde kullanılan advanced_features.csv ile uyumlu olmalı.
  Yanıt: toplam satır, anomali sayısı/oranı ve skor istatistikleri.
  """
  import io

  if not file.filename.lower().endswith(".csv"):
      raise HTTPException(status_code=400, detail="Lütfen .csv uzantılı bir dosya yükleyin")

  raw = await file.read()
  if not raw:
      raise HTTPException(status_code=400, detail="Dosya boş")

  try:
      import pandas as pd
      import numpy as np
  except Exception as e:
      raise HTTPException(status_code=500, detail=f"Pandas veya Numpy yüklenemedi: {e}")

  try:
      text = raw.decode("utf-8", errors="ignore")
      df = pd.read_csv(io.StringIO(text))
  except Exception:
      raise HTTPException(status_code=400, detail="CSV dosyası okunamadı")

  if df.empty:
      raise HTTPException(status_code=400, detail="CSV dosyası satır içermiyor")

  detector, feature_names, model_name = _get_detector(model)
  if detector is None or feature_names is None or model_name == "demo":
      raise HTTPException(status_code=400, detail="İstenen model sunucuda yüklü değil")

  # Kolonları modelin beklediği sıraya göre hizala; eksik kolonları 0.0 ile doldur.
  missing = [c for c in feature_names if c not in df.columns]
  if len(missing) == len(feature_names):
      raise HTTPException(
          status_code=400,
          detail="CSV kolonları eğitimde kullanılan feature set'i ile uyuşmuyor",
      )

  df_features = df.reindex(columns=feature_names, fill_value=0.0)
  X = df_features.values.astype("float32")

  try:
      if model_name == "lstm_ae":
          # sequence_length=1 varsayıyoruz
          n_cols = X.shape[1]
          X = X.reshape(-1, 1, n_cols)

      preds, scores = detector.predict(X)
      preds_arr = np.array(preds).astype(int)
      scores_arr = np.array(scores, dtype="float32").reshape(-1)
  except Exception as e:
      raise HTTPException(status_code=500, detail=f"Model tahmini sırasında hata: {e}")

  total = int(len(scores_arr))
  anomalies = int((preds_arr == 1).sum())
  ratio = float(anomalies / total) if total > 0 else 0.0

  score_min = float(scores_arr.min()) if total > 0 else 0.0
  score_max = float(scores_arr.max()) if total > 0 else 0.0
  score_mean = float(scores_arr.mean()) if total > 0 else 0.0

  return {
      "model": model_name,
      "total_rows": total,
      "anomaly_count": anomalies,
      "anomaly_ratio": ratio,
      "score_min": score_min,
      "score_max": score_max,
      "score_mean": score_mean,
  }


@app.post("/api/analyze-pcap")
async def analyze_pcap(file: UploadFile = File(...), model: Optional[str] = Form(None)):
    """
    Kullanıcının yüklediği PCAP dosyasını seçilen modelle toplu analiz eder.
    PCAP, basit flow istatistiklerine dönüştürülür; kolonlar advanced_features ile hizalanır.
    """
    if not (file.filename.lower().endswith(".pcap") or file.filename.lower().endswith(".pcapng")):
        raise HTTPException(status_code=400, detail="Lütfen .pcap veya .pcapng uzantılı bir dosya yükleyin")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Dosya boş")

    try:
        import numpy as np
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Numpy yüklenemedi: {e}")

    detector, feature_names, model_name = _get_detector(model)
    if detector is None or feature_names is None or model_name == "demo":
        raise HTTPException(status_code=400, detail="İstenen model sunucuda yüklü değil")

    try:
        X = _pcap_to_feature_rows(raw, feature_names)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PCAP okunamadı veya işlenemedi: {e}")

    if X is None or len(X) == 0:
        raise HTTPException(status_code=400, detail="PCAP dosyasından geçerli flow çıkarılamadı")

    try:
        if model_name == "lstm_ae":
            n_cols = X.shape[1]
            X = X.reshape(-1, 1, n_cols)

        preds, scores = detector.predict(X)
        preds_arr = np.array(preds).astype(int)
        scores_arr = np.array(scores, dtype="float32").reshape(-1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model tahmini sırasında hata: {e}")

    total = int(len(scores_arr))
    anomalies = int((preds_arr == 1).sum())
    ratio = float(anomalies / total) if total > 0 else 0.0

    score_min = float(scores_arr.min()) if total > 0 else 0.0
    score_max = float(scores_arr.max()) if total > 0 else 0.0
    score_mean = float(scores_arr.mean()) if total > 0 else 0.0

    return {
        "model": model_name,
        "total_rows": total,
        "anomaly_count": anomalies,
        "anomaly_ratio": ratio,
        "score_min": score_min,
        "score_max": score_max,
        "score_mean": score_mean,
    }
