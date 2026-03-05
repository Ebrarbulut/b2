"""
📥 UNSW-NB15 VERİ SETİ YÜKLEYİCİ
==================================

UNSW-NB15 dataset'ini yükler ve conn.log formatına dönüştürür.

KULLANIM:
from datasets.unsw_nb15_loader import UNSWNB15Loader
loader = UNSWNB15Loader()
df = loader.load_and_convert('UNSW_NB15_training-set.csv')
"""

import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
import zipfile
import os

class UNSWNB15Loader:
    """
    UNSW-NB15 dataset loader ve converter
    """
    
    def __init__(self, data_dir='data/raw'):
        """
        Args:
            data_dir: Veri seti dizini
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # UNSW-NB15 feature mapping (conn.log formatına)
        self.feature_mapping = {
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
    
    def download_dataset(self, url=None, output_file='unsw_nb15.zip'):
        """
        UNSW-NB15 dataset'ini indir
        
        Not: Manuel indirme gerekebilir (link verilir)
        """
        if url is None:
            url = "https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/"
            print(f"⚠️  Manuel indirme gerekli:")
            print(f"   {url}")
            print(f"   Dosyaları {self.data_dir} klasörüne koyun")
            return False
        
        output_path = self.data_dir / output_file
        
        print(f"📥 Downloading UNSW-NB15 from {url}...")
        try:
            urllib.request.urlretrieve(url, output_path)
            print(f"✅ Downloaded: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def load_csv(self, csv_file):
        """
        UNSW-NB15 CSV dosyasını yükle
        
        Args:
            csv_file: CSV dosya yolu
        """
        csv_path = Path(csv_file)
        if not csv_path.exists():
            csv_path = self.data_dir / csv_file
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        print(f"📂 Loading {csv_path}...")
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"✅ Loaded {len(df)} records")
        
        return df
    
    def convert_to_conn_log_format(self, df):
        """
        UNSW-NB15 formatını conn.log formatına dönüştür
        
        Args:
            df: UNSW-NB15 DataFrame
        """
        print("🔄 Converting to conn.log format...")
        
        # Yeni DataFrame oluştur
        conn_log = pd.DataFrame()
        
        # Mapping yap
        for unsw_col, conn_col in self.feature_mapping.items():
            if unsw_col in df.columns:
                conn_log[conn_col] = df[unsw_col]
            else:
                print(f"⚠️  Column '{unsw_col}' not found, using default")
                conn_log[conn_col] = 0
        
        # Timestamp ekle (eğer yoksa)
        if 'ts' not in conn_log.columns:
            if 'stime' in df.columns:
                conn_log['ts'] = pd.to_datetime(df['stime'])
            else:
                # Sequential timestamp
                conn_log['ts'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1S')
        
        # Label ekle
        if 'label' in df.columns:
            conn_log['label'] = df['label'].map({0: 'normal', 1: 'anomaly'})
        elif 'attack_cat' in df.columns:
            conn_log['label'] = df['attack_cat'].apply(
                lambda x: 'anomaly' if pd.notna(x) and x != 'Normal' else 'normal'
            )
            conn_log['attack_type'] = df['attack_cat'].fillna('normal')
        else:
            conn_log['label'] = 'normal'
        
        # Eksik kolonları doldur
        required_cols = [
            'uid', 'local_orig', 'local_resp', 'missed_bytes',
            'history', 'orig_ip_bytes', 'resp_ip_bytes', 'tunnel_parents'
        ]
        
        for col in required_cols:
            if col not in conn_log.columns:
                conn_log[col] = '-'
        
        print(f"✅ Converted {len(conn_log)} records")
        print(f"   Normal: {(conn_log['label'] == 'normal').sum()}")
        print(f"   Anomaly: {(conn_log['label'] == 'anomaly').sum()}")
        
        return conn_log
    
    def save_conn_log(self, conn_log_df, output_file='conn.log'):
        """
        conn.log formatında kaydet
        
        Args:
            conn_log_df: conn.log formatında DataFrame
            output_file: Çıktı dosya adı
        """
        output_path = self.data_dir / output_file
        
        # Tab-separated format (Zeek conn.log formatı)
        conn_log_df.to_csv(
            output_path,
            sep='\t',
            index=False,
            header=False  # Zeek formatında header yok
        )
        
        print(f"✅ Saved conn.log: {output_path}")
        return output_path
    
    def load_and_convert(self, csv_file, save_conn_log=True):
        """
        CSV'yi yükle ve conn.log formatına dönüştür
        
        Args:
            csv_file: CSV dosya yolu
            save_conn_log: conn.log olarak kaydet
        """
        # Yükle
        df = self.load_csv(csv_file)
        
        # Dönüştür
        conn_log_df = self.convert_to_conn_log_format(df)
        
        # Kaydet
        if save_conn_log:
            self.save_conn_log(conn_log_df)
        
        return conn_log_df


# =============================================================================
# KULLANIM ÖRNEĞİ
# =============================================================================

if __name__ == "__main__":
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║            UNSW-NB15 DATASET LOADER                       ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    loader = UNSWNB15Loader()
    
    # CSV dosyası yolu (kullanıcı kendi dosyasını belirtmeli)
    csv_file = input("UNSW-NB15 CSV dosya yolu: ").strip()
    
    if csv_file:
        # Yükle ve dönüştür
        conn_log_df = loader.load_and_convert(csv_file)
        print("\n✅ UNSW-NB15 dataset hazır!")
        print("   Şimdi 'python scripts/label_traffic.py' çalıştırabilirsiniz")
    else:
        print("\n💡 Kullanım:")
        print("   1. UNSW-NB15 dataset'ini indirin:")
        print("      https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/")
        print("   2. CSV dosyasını data/raw/ klasörüne koyun")
        print("   3. Bu scripti tekrar çalıştırın")
