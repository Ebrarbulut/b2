"""
📥 HAZIR VERİ SETLERİNİ İNDİRME
=================================

Bu script, test için hazır anomali tespiti veri setlerini indirir.

KULLANIM:
    python scripts/download_datasets.py
"""

import urllib.request
import zipfile
import tarfile
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("📥 HAZIR VERİ SETLERİNİ İNDİRME")
print("=" * 70)

# === SEÇENEK 1: UNSW-NB15 (Önerilen) ===
print("\n📊 UNSW-NB15 Dataset")
print("-" * 70)
print("Bu veri seti, gerçekçi normal ve anomali trafiği içerir.")
print("İndirme linki: https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/")
print("\n⚠️ Manuel indirme gerekli:")
print("1. Yukarıdaki linkten UNSW-NB15 dataset'ini indirin")
print("2. conn.log formatına dönüştürün veya CSV formatında kullanın")
print("3. data/raw/ klasörüne yerleştirin")

# === SEÇENEK 2: CICIDS2017 ===
print("\n📊 CICIDS2017 Dataset")
print("-" * 70)
print("Bu veri seti, çeşitli saldırı türlerini içerir.")
print("İndirme linki: https://www.unb.ca/cic/datasets/ids-2017.html")
print("\n⚠️ Manuel indirme gerekli:")
print("1. Yukarıdaki linkten CICIDS2017 dataset'ini indirin")
print("2. PCAP dosyalarını Zeek ile işleyin: zeek -r file.pcap")
print("3. conn.log dosyasını data/raw/ klasörüne yerleştirin")

# === SEÇENEK 3: KDD Cup 1999 (Küçük test için) ===
print("\n📊 KDD Cup 1999 Dataset (Test için)")
print("-" * 70)
print("Bu veri seti, küçük testler için uygundur.")
print("İndirme linki: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html")

kdd_url = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
kdd_file = DATA_DIR / "kddcup.data_10_percent.gz"

print(f"\n💡 KDD Cup 1999 indiriliyor...")
try:
    urllib.request.urlretrieve(kdd_url, kdd_file)
    print(f"✅ İndirildi: {kdd_file}")
    print("⚠️ Bu dosya Zeek conn.log formatında değil, dönüştürme gerekebilir.")
except Exception as e:
    print(f"❌ İndirme hatası: {e}")

# === ÖNERİLEN YOL ===
print("\n" + "=" * 70)
print("💡 ÖNERİLEN YOL")
print("=" * 70)
print("""
1. Kendi Zeek loglarınızı kullanın (data/raw/conn.log)
2. Eğer yoksa, gerçekçi test için:
   - UNSW-NB15 veya CICIDS2017 dataset'lerini indirin
   - Zeek ile işleyin: zeek -r traffic.pcap
   - conn.log dosyasını data/raw/ klasörüne kopyalayın

3. Test için:
   - python scripts/label_traffic.py  (veriyi etiketle)
   - python scripts/advanced_features.py  (feature engineering)
   - python scripts/test_pipeline.py  (tam test)
""")

print("\n✅ Hazır!")
