"""
Basit canlı analiz ajanı.

Amaç:
- Yerel bir CSV/log dosyasına eklenen yeni satırları periyodik olarak oku.
- Bu satırlardaki sayısal feature'ları backend /api/score endpoint'ine gönder.
- Gelen anomali skorlarını terminalde "normal / şüpheli / anomali" şeklinde göster.

Kullanım örneği:
  python scripts/live_log_agent.py ^
      --csv-path data/features/advanced_features.csv ^
      --backend-url http://127.0.0.1:8000

Not:
- CSV'deki kolonlar backend'de kullanılan modelin beklediği feature sayısıyla
  uyumlu olmalıdır (ör: advanced_features.csv).
- Gerçek makine trafiğini analiz etmek için, kendi ağ loglarını bu formatta
  CSV'ye dönüştürüp bu script'e verebilirsin.
"""

import argparse
import time
from typing import List

import numpy as np
import pandas as pd
import requests


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Canlı log ajanı (CSV -> /api/score).")
  parser.add_argument(
      "--csv-path",
      type=str,
      required=True,
      help="Takip edilecek CSV dosyasının yolu.",
  )
  parser.add_argument(
      "--backend-url",
      type=str,
      default="http://127.0.0.1:8000",
      help="FastAPI backend base URL (varsayılan: http://127.0.0.1:8000).",
  )
  parser.add_argument(
      "--interval",
      type=int,
      default=10,
      help="Saniye cinsinden okuma/analiz aralığı (varsayılan: 10).",
  )
  parser.add_argument(
      "--batch-size",
      type=int,
      default=32,
      help="Her turda analiz edilecek maksimum yeni satır sayısı (varsayılan: 32).",
  )
  return parser.parse_args()


def select_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
  """Sadece sayısal kolonları seç."""
  numeric_df = df.select_dtypes(include=["number"])
  return numeric_df


def fetch_scores(backend_url: str, features: List[List[float]]) -> List[float]:
  payload = {"features": features}
  resp = requests.post(f"{backend_url}/api/score", json=payload, timeout=10)
  resp.raise_for_status()
  data = resp.json()
  scores = data.get("scores", [])
  return [float(s) for s in scores]


def describe_scores(scores: List[float]) -> str:
  if not scores:
    return "Hiç skor yok."

  arr = np.array(scores, dtype=float)
  mean_score = float(np.mean(arr))

  if mean_score < 0.3:
    level = "normal"
  elif mean_score < 0.7:
    level = "şüpheli"
  else:
    level = "anomalik"

  return f"Ortalama skor: {mean_score:.3f} → durum: {level}"


def main() -> None:
  args = parse_args()
  csv_path = args.csv_path
  backend_url = args.backend_url.rstrip("/")
  interval = args.interval
  batch_size = args.batch_size

  print(f"[agent] CSV: {csv_path}")
  print(f"[agent] Backend: {backend_url}")
  print(f"[agent] Interval: {interval}s, batch size: {batch_size}")

  last_row = 0

  while True:
    try:
      df = pd.read_csv(csv_path)
      if df.empty:
        print("[agent] CSV boş, bekleniyor...")
      else:
        if last_row >= len(df):
          print("[agent] Yeni satır yok, bekleniyor...")
        else:
          new_df = df.iloc[last_row : last_row + batch_size]
          last_row += len(new_df)
          numeric_df = select_numeric_features(new_df)
          features = numeric_df.values.tolist()
          try:
            scores = fetch_scores(backend_url, features)
            summary = describe_scores(scores)
            print(f"[agent] Yeni {len(features)} kayıt analiz edildi. {summary}")
          except Exception as e:  # noqa: BLE001
            print(f"[agent] Skor isteği hatası: {e}")
    except FileNotFoundError:
      print(f"[agent] CSV bulunamadı: {csv_path}")
    except Exception as e:  # noqa: BLE001
      print(f"[agent] Okuma/analiz hatası: {e}")

    time.sleep(interval)


if __name__ == "__main__":
  main()

