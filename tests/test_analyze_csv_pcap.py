"""
/api/analyze-csv ve /api/analyze-pcap endpoint'leri için testler.
"""
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.main import app

client = TestClient(app)


def test_analyze_csv_wrong_extension():
    """CSV yerine .txt gönderilirse 400 dönmeli."""
    response = client.post(
        "/api/analyze-csv",
        files={"file": ("data.txt", b"a,b,c\n1,2,3", "text/plain")},
        data={},
    )
    assert response.status_code == 400
    assert "csv" in response.json().get("detail", "").lower()


def test_analyze_csv_empty_file():
    """Boş dosya gönderilirse 400 dönmeli."""
    response = client.post(
        "/api/analyze-csv",
        files={"file": ("data.csv", b"", "text/csv")},
        data={},
    )
    assert response.status_code == 400
    assert "boş" in response.json().get("detail", "").lower()


def test_analyze_csv_invalid_csv():
    """Geçersiz CSV içeriği 400 dönmeli."""
    response = client.post(
        "/api/analyze-csv",
        files={"file": ("data.csv", b"not,csv,content\nbroken", "text/csv")},
        data={},
    )
    # Okunamadı veya kolon uyuşmazlığı / model yok
    assert response.status_code in (400, 500)


def test_analyze_pcap_wrong_extension():
    """PCAP yerine .txt gönderilirse 400 dönmeli."""
    response = client.post(
        "/api/analyze-pcap",
        files={"file": ("data.txt", b"fake", "application/octet-stream")},
        data={},
    )
    assert response.status_code == 400
    assert "pcap" in response.json().get("detail", "").lower()


def test_analyze_pcap_empty_file():
    """Boş PCAP dosyası 400 dönmeli."""
    response = client.post(
        "/api/analyze-pcap",
        files={"file": ("capture.pcap", b"", "application/vnd.tcpdump.pcap")},
        data={},
    )
    assert response.status_code == 400
    assert "boş" in response.json().get("detail", "").lower()
