"""
🔗 LOG BİRLEŞTİRME VE OTOMATİK ETİKETLEME
==========================================

Bu script, data/pcap/logs_* klasörlerindeki tüm conn.log dosyalarını
birleştirir ve klasör adına göre otomatik etiketler.

KULLANIM:
    python scripts/merge_and_label_logs.py
"""

import pandas as pd
from pathlib import Path
import sys

# Proje kök dizinini sys.path'e ekle ki `scripts.*` paketleri her yerden import edilebilsin
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.config.paths import BASE_DIR, PCAP_DIR, RAW_DIR, LABELED_DIR

# Windows console encoding fix
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

print("=" * 70)
print("🔗 LOG BİRLEŞTİRME VE OTOMATİK ETİKETLEME")
print("=" * 70)

RAW_DIR.mkdir(parents=True, exist_ok=True)
LABELED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CONN_LOG = RAW_DIR / "conn.log"
OUTPUT_LABELED = LABELED_DIR / "labeled_traffic.csv"

# === ETİKET MAPPING ===
# Klasör adına göre saldırı tipi belirleme
LABEL_MAPPING = {
    "logs_normal": ("normal", "normal"),
    "logs_syn": ("anomaly", "syn_flood"),
    "logs_dns": ("anomaly", "dns_amplification"),
    "logs_http": ("anomaly", "http_flood"),
    "logs_udp": ("anomaly", "udp_flood"),
}

# === ZEEK KOLONLARI ===
ZEEK_COLUMNS = [
    'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p',
    'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes',
    'conn_state', 'local_orig', 'local_resp', 'missed_bytes',
    'history', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes',
    'tunnel_parents', 'ip_proto'
]

# === LOG KLASÖRLERINI BUL ===
log_dirs = list(PCAP_DIR.glob("logs_*"))

if not log_dirs:
    print("❌ data/pcap/ altında logs_* klasörleri bulunamadı!")
    sys.exit(1)

print(f"\n📂 Bulunan log klasörleri: {len(log_dirs)}")
for log_dir in log_dirs:
    print(f"   - {log_dir.name}")

# === TÜM LOGLARI BİRLEŞTİR ===
all_data = []
total_lines = 0

for log_dir in log_dirs:
    conn_log = log_dir / "conn.log"
    
    if not conn_log.exists():
        print(f"⚠️  {log_dir.name}/conn.log bulunamadı, atlanıyor...")
        continue
    
    # Klasör adından etiket belirle
    folder_name = log_dir.name
    if folder_name in LABEL_MAPPING:
        label, attack_type = LABEL_MAPPING[folder_name]
    else:
        print(f"⚠️  {folder_name} için etiket tanımlı değil, 'normal' kabul ediliyor")
        label, attack_type = "normal", "normal"
    
    print(f"\n📄 İşleniyor: {folder_name}")
    print(f"   Etiket: {label} ({attack_type})")
    
    try:
        # Zeek log'u oku (# ile başlayan satırları atla)
        df = pd.read_csv(
            conn_log,
            sep="\t",
            comment="#",
            names=ZEEK_COLUMNS,
            on_bad_lines='skip',
            low_memory=False
        )
        
        # Etiket ekle
        df['label'] = label
        df['attack_type'] = attack_type
        
        print(f"   ✅ {len(df)} satır yüklendi")
        total_lines += len(df)
        
        all_data.append(df)
        
    except Exception as e:
        print(f"   ❌ Hata: {e}")
        continue

if not all_data:
    print("\n❌ Hiçbir log dosyası işlenemedi!")
    sys.exit(1)

# === BİRLEŞTİR ===
print(f"\n🔗 Tüm loglar birleştiriliyor...")
combined_df = pd.concat(all_data, ignore_index=True)

print(f"✅ Toplam {len(combined_df)} satır birleştirildi")
print(f"\n📊 Etiket Dağılımı:")
print(combined_df['label'].value_counts())
print(f"\n📊 Saldırı Tipi Dağılımı:")
print(combined_df['attack_type'].value_counts())

# === KAYDET ===
print(f"\n💾 Kaydediliyor...")

# 1. Etiketli CSV olarak kaydet (label_traffic.py'nin çıktısı gibi)
combined_df.to_csv(OUTPUT_LABELED, index=False)
print(f"✅ Etiketli veri kaydedildi: {OUTPUT_LABELED}")

# 2. Ham conn.log formatında da kaydet (opsiyonel, ileride kullanılabilir)
# Etiket kolonlarını çıkar ve Zeek formatında kaydet
conn_only = combined_df[ZEEK_COLUMNS]
with open(OUTPUT_CONN_LOG, 'w') as f:
    # Zeek header ekle
    f.write("#separator \\x09\n")
    f.write("#set_separator\t,\n")
    f.write("#empty_field\t(empty)\n")
    f.write("#unset_field\t-\n")
    f.write("#path\tconn\n")
    f.write(f"#fields\t" + "\t".join(ZEEK_COLUMNS) + "\n")
    f.write("#types\ttime\tstring\taddr\tport\taddr\tport\tenum\tstring\tinterval\tcount\tcount\tstring\tbool\tbool\tcount\tstring\tcount\tcount\tcount\tcount\tset[string]\tcount\n")
    
    # Veriyi yaz
    conn_only.to_csv(f, sep='\t', header=False, index=False)
    f.write("#close\t2026-02-07-20-54-48\n")

print(f"✅ Ham conn.log kaydedildi: {OUTPUT_CONN_LOG}")

# === ÖZET ===
print("\n" + "=" * 70)
print("✅ BİRLEŞTİRME TAMAMLANDI!")
print("=" * 70)
print(f"\n📁 Çıktı dosyaları:")
print(f"   1. {OUTPUT_LABELED}")
print(f"   2. {OUTPUT_CONN_LOG}")
print(f"\n📊 Veri özeti:")
print(f"   - Toplam kayıt: {len(combined_df)}")
print(f"   - Normal: {(combined_df['label'] == 'normal').sum()}")
print(f"   - Anomaly: {(combined_df['label'] == 'anomaly').sum()}")
print(f"\n🚀 Sonraki adımlar:")
print(f"   1. python scripts/data_pipeline/advanced_features.py")
print(f"   2. python scripts/experiements/compare_all_models.py")
print(f"   3. streamlit run streamlit_app.py")
