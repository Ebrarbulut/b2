import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Windows console encoding fix (avoid UnicodeEncodeError for emojis / Turkish chars)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

print("=" * 60)
print("TRAFFIC LABELING")
print("=" * 60)

# === PATHS ===
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
LABELED_DIR = BASE_DIR / "data" / "labeled"
SCRIPTS_DIR = BASE_DIR / "scripts"

LABELED_DIR.mkdir(parents=True, exist_ok=True)

# Scripts dizinini path'e ekle
sys.path.insert(0, str(SCRIPTS_DIR))

CONN_LOG = RAW_DIR / "conn.log"
ATTACK_LOG = RAW_DIR / "attack_log.csv"
UNSW_CSV = RAW_DIR / "UNSW_NB15_training-set.csv"
OUTPUT_FILE = LABELED_DIR / "labeled_traffic.csv"

# === CHECK UNSW-NB15 CSV FIRST ===
if UNSW_CSV.exists():
    print("📂 UNSW-NB15 CSV bulundu, direkt CSV'den yükleniyor...")
    
    # UNSW-NB15 CSV'den direkt yükle
    df = pd.read_csv(UNSW_CSV, low_memory=False)
    print(f"✅ Loaded {len(df)} records from UNSW-NB15 CSV")
    
    # UNSW-NB15 formatını conn.log formatına dönüştür
    try:
        from datasets.unsw_nb15_loader import UNSWNB15Loader
        loader = UNSWNB15Loader()
        df = loader.convert_to_conn_log_format(df)
        print("✅ Label bilgileri CSV'den alındı")
    except ImportError as e:
        print(f"⚠️  Import hatası: {e}")
        print("🔄 Manuel dönüştürme yapılıyor...")
        
        # Manuel dönüştürme
        conn_log = pd.DataFrame()
        
        # Feature mapping
        mapping = {
            'dur': 'duration',
            'sbytes': 'orig_bytes',
            'dbytes': 'resp_bytes',
            'spkts': 'orig_pkts',
            'dpkts': 'resp_pkts',
            'sport': 'id.orig_p',
            'dport': 'id.resp_p',
            'srcip': 'id.orig_h',
            'dstip': 'id.resp_h',
            'proto': 'proto',
            'state': 'conn_state',
            'service': 'service'
        }
        
        for unsw_col, conn_col in mapping.items():
            if unsw_col in df.columns:
                conn_log[conn_col] = df[unsw_col]
            else:
                conn_log[conn_col] = 0
        
        # Timestamp
        conn_log['ts'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1S')
        
        # Label ekle
        if 'label' in df.columns:
            conn_log['label'] = df['label'].map({0: 'normal', 1: 'anomaly'})
        elif 'attack_cat' in df.columns:
            conn_log['label'] = df['attack_cat'].apply(
                lambda x: 'anomaly' if pd.notna(x) and str(x) != 'Normal' else 'normal'
            )
            conn_log['attack_type'] = df['attack_cat'].fillna('normal')
        else:
            conn_log['label'] = 'normal'
            conn_log['attack_type'] = 'normal'
        
        # Eksik kolonları doldur
        required_cols = ['uid', 'local_orig', 'local_resp', 'missed_bytes',
                        'history', 'orig_ip_bytes', 'resp_ip_bytes', 'tunnel_parents']
        for col in required_cols:
            if col not in conn_log.columns:
                conn_log[col] = '-'
        
        df = conn_log
        print(f"✅ Converted {len(df)} records")
        print(f"   Normal: {(df['label'] == 'normal').sum()}")
        print(f"   Anomaly: {(df['label'] == 'anomaly').sum()}")
    
elif CONN_LOG.exists():
    print("📂 Loading Zeek conn.log...")
    
    columns = [
        'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p',
        'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes',
        'conn_state', 'local_orig', 'local_resp', 'missed_bytes',
        'history', 'orig_pkts', 'orig_ip_bytes',
        'resp_pkts', 'resp_ip_bytes', 'tunnel_parents'
    ]
    
    df = pd.read_csv(
        CONN_LOG,
        sep="\t",
        comment="#",
        names=columns,
        low_memory=False
    )
    
    print(f"✅ Loaded {len(df)} raw connections")
else:
    raise FileNotFoundError(f"❌ Ne conn.log ne de UNSW-NB15 CSV bulunamadı!")

# === FIX TIMESTAMP (KRİTİK) ===
# Eğer timestamp zaten datetime ise, numeric'e çevirmeye çalışma
if df['ts'].dtype == 'object' or not pd.api.types.is_datetime64_any_dtype(df['ts']):
    try:
        # Önce numeric'e çevir (Unix timestamp ise)
        df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
        invalid_ts = df['ts'].isna().sum()
        df = df.dropna(subset=['ts'])
        if invalid_ts > 0:
            print(f"🧹 Dropped {invalid_ts} invalid timestamp rows")
        df['ts'] = pd.to_datetime(df['ts'], unit='s')
    except:
        # Zaten datetime formatında olabilir
        df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
        invalid_ts = df['ts'].isna().sum()
        df = df.dropna(subset=['ts'])
        if invalid_ts > 0:
            print(f"🧹 Dropped {invalid_ts} invalid timestamp rows")
else:
    # Zaten datetime
    invalid_ts = df['ts'].isna().sum()
    df = df.dropna(subset=['ts'])
    if invalid_ts > 0:
        print(f"🧹 Dropped {invalid_ts} invalid timestamp rows")

# === LABELS ===
# Eğer UNSW-NB15'ten geldiyse label zaten var
if 'label' not in df.columns:
    df['label'] = 'normal'
    df['attack_type'] = 'normal'
    
    # === LOAD attack_log.csv (VARSA) ===
    if ATTACK_LOG.exists():
        print("📂 Loading attack_log.csv...")
        attacks = pd.read_csv(ATTACK_LOG)

        attacks['start_time'] = pd.to_datetime(attacks['start_time'])
        attacks['end_time'] = pd.to_datetime(attacks['end_time'])

        for _, attack in attacks.iterrows():
            mask = (df['ts'] >= attack['start_time']) & (df['ts'] <= attack['end_time'])
            df.loc[mask, 'label'] = 'anomaly'
            df.loc[mask, 'attack_type'] = attack['attack_type']
    else:
        print("⚠️ attack_log.csv bulunamadı → tüm trafik NORMAL kabul edildi")
else:
    # Label zaten var, attack_type kontrol et
    if 'attack_type' not in df.columns:
        df['attack_type'] = df.get('label', 'normal')
    print("✅ Label bilgileri CSV'den alındı")

# === SAVE ===
df.to_csv(OUTPUT_FILE, index=False)

print(f"💾 Saved: {OUTPUT_FILE}")
print("\n📊 Label Distribution:")
print(df['label'].value_counts())
