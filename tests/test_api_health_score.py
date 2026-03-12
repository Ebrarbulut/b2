import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Proje kökünü sys.path'e ekle ki `backend` modülü bulunabilsin
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.main import app


client = TestClient(app)


def test_health_endpoint():
  """API /health endpoint'i 200 ve beklenen alanları döndürmeli."""
  response = client.get("/health")
  assert response.status_code == 200
  data = response.json()
  assert data.get("status") == "ok"
  assert data.get("service") == "nad-api"


def test_score_endpoint_demo_mode():
  """Model olmasa bile /api/score demo skorları döndürmeli."""
  payload = {
      "features": [
          [0, 100, 200, 5, 10, 300, 15, 0.5, 0.5, 0.1, 0.2, 2]
      ]
  }
  response = client.post("/api/score", json=payload)
  assert response.status_code == 200
  data = response.json()
  assert "scores" in data
  assert isinstance(data["scores"], list)
  assert len(data["scores"]) == 1

