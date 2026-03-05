import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import entropy
import sys

# Windows console encoding fix (avoid UnicodeEncodeError for emojis / Turkish chars)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

print("🚀 Advanced Feature Engineering")

# === PATHS ===
BASE_DIR = Path.cwd()  # Use current working directory instead of __file__
LABELED_FILE = BASE_DIR / "data" / "labeled" / "labeled_traffic.csv"
FEATURE_DIR = BASE_DIR / "data" / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = FEATURE_DIR / "advanced_features.csv"

# === LOAD ===
if not LABELED_FILE.exists():
    raise FileNotFoundError(f"❌ labeled_traffic.csv yok: {LABELED_FILE}")

df = pd.read_csv(LABELED_FILE)

if df.empty:
    raise ValueError("❌ DataFrame boş. Labeling başarısız.")

print(f"✅ Loaded {len(df)} labeled connections")

# === FIX TIMESTAMP (CSV'den string olarak gelir) ===
if 'ts' in df.columns:
    # Önce numeric'e çevir (eğer Unix timestamp ise)
    df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
    # Sonra datetime'a çevir
    df['ts'] = pd.to_datetime(df['ts'], unit='s', errors='coerce')
    # Eğer başarısız olursa, direkt parse et
    if df['ts'].isna().any():
        df['ts'] = pd.to_datetime(df['ts'], errors='coerce')

# === BASIC NUMERIC CLEAN ===
numeric_cols = [
    'duration', 'orig_bytes', 'resp_bytes',
    'orig_pkts', 'resp_pkts'
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# === FEATURE ENGINEERING ===
df['total_bytes'] = df['orig_bytes'] + df['resp_bytes']
df['total_pkts'] = df['orig_pkts'] + df['resp_pkts']

df['bytes_ratio'] = np.where(df['resp_bytes'] > 0,
                              df['orig_bytes'] / df['resp_bytes'], 0)

df['pkts_ratio'] = np.where(df['resp_pkts'] > 0,
                             df['orig_pkts'] / df['resp_pkts'], 0)

# Port entropy hesaplama (normalize edilmiş value_counts gerekli)
def calc_entropy(series):
    """Port entropy hesapla"""
    if len(series) <= 1:
        return 0
    counts = series.value_counts()
    probs = counts / counts.sum()
    return entropy(probs, base=2)

if 'id.orig_h' in df.columns and 'id.orig_p' in df.columns:
    df['src_port_entropy'] = df.groupby('id.orig_h')['id.orig_p'].transform(calc_entropy)
else:
    df['src_port_entropy'] = 0

if 'id.orig_h' in df.columns and 'id.resp_p' in df.columns:
    df['dst_port_entropy'] = df.groupby('id.orig_h')['id.resp_p'].transform(calc_entropy)
else:
    df['dst_port_entropy'] = 0

# Connections per minute hesaplama
if 'ts' in df.columns and 'id.orig_h' in df.columns:
    # ts datetime olmalı
    if df['ts'].dtype == 'object':
        df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
    
    # NaN timestamp'leri temizle
    df_with_ts = df.dropna(subset=['ts'])
    
    if not df_with_ts.empty:
        df['connections_per_min'] = (
            df_with_ts.groupby(['id.orig_h', pd.Grouper(key='ts', freq='1min')])
            .transform('size')
        )
        # NaN değerleri 0 ile doldur
        df['connections_per_min'] = df['connections_per_min'].fillna(0)
    else:
        df['connections_per_min'] = 0
else:
    df['connections_per_min'] = 0

# === FINAL FEATURE SET ===
features = [
    'duration', 'orig_bytes', 'resp_bytes',
    'orig_pkts', 'resp_pkts',
    'total_bytes', 'total_pkts',
    'bytes_ratio', 'pkts_ratio',
    'src_port_entropy', 'dst_port_entropy',
    'connections_per_min'
]

# Sadece mevcut feature'ları seç
available_features = [f for f in features if f in df.columns]

if not available_features:
    raise ValueError("❌ Hiçbir feature bulunamadı!")

print(f"📊 Using {len(available_features)} features: {', '.join(available_features)}")

X = df[available_features].fillna(0)

if X.empty:
    raise ValueError("❌ Feature matrix boş. Veri üretilemedi.")

X.to_csv(OUTPUT_FILE, index=False)

print(f"💾 Saved: {OUTPUT_FILE}")
print(f"📐 Feature matrix shape: {X.shape}")
