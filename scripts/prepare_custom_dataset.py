"""
🔧 KENDİ VERİ SETİNİZİ HAZIRLAMA SİSTEMİ
=========================================

PCAP dosyasından conn.log çıkarma ve veri hazırlama.

KULLANIM:
python scripts/prepare_custom_dataset.py
"""

import subprocess
import sys
from pathlib import Path
import os

def check_zeek_installation():
    """
    Zeek kurulumunu kontrol et
    """
    try:
        result = subprocess.run(
            ['zeek', '--version'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ Zeek kurulu")
            print(f"   {result.stdout.split()[0]}")
            return True
        else:
            print("❌ Zeek bulunamadı")
            return False
    except FileNotFoundError:
        print("❌ Zeek bulunamadı")
        return False

def pcap_to_conn_log(pcap_file, output_dir='data/raw'):
    """
    PCAP dosyasını Zeek ile işle ve conn.log çıkar
    
    Args:
        pcap_file: PCAP dosya yolu
        output_dir: Çıktı dizini
    """
    pcap_path = Path(pcap_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not pcap_path.exists():
        print(f"❌ PCAP dosyası bulunamadı: {pcap_path}")
        return None
    
    print(f"🔄 Processing {pcap_path} with Zeek...")
    
    # Zeek komutu
    cmd = ['zeek', '-r', str(pcap_path), '-C']
    
    try:
        # Zeek'i çalıştır
        result = subprocess.run(
            cmd,
            cwd=str(output_path),
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            conn_log_path = output_path / 'conn.log'
            if conn_log_path.exists():
                print(f"✅ conn.log oluşturuldu: {conn_log_path}")
                return conn_log_path
            else:
                print("⚠️  conn.log oluşturulamadı")
                return None
        else:
            print(f"❌ Zeek hatası: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Hata: {e}")
        return None

def main():
    """
    Ana fonksiyon
    """
    print("=" * 70)
    print("🔧 KENDİ VERİ SETİNİZİ HAZIRLAMA")
    print("=" * 70)
    
    # Zeek kontrolü
    print("\n1️⃣ Checking Zeek installation...")
    if not check_zeek_installation():
        print("\n💡 Zeek kurulumu için:")
        print("   Windows: choco install zeek")
        print("   Linux: sudo apt-get install zeek")
        print("   Mac: brew install zeek")
        return
    
    # PCAP dosyası sor
    print("\n2️⃣ PCAP dosyası seçin:")
    pcap_file = input("PCAP dosya yolu (veya Enter ile atla): ").strip()
    
    if pcap_file:
        conn_log_path = pcap_to_conn_log(pcap_file)
        
        if conn_log_path:
            print(f"\n✅ Veri seti hazır!")
            print(f"   conn.log: {conn_log_path}")
            print("\n📝 Sonraki adımlar:")
            print("   1. python scripts/label_traffic.py")
            print("   2. python scripts/advanced_features.py")
            print("   3. python scripts/compare_all_models.py")
    else:
        print("\n💡 Kullanım:")
        print("   1. PCAP dosyanızı hazırlayın (Wireshark ile yakalayın)")
        print("   2. Bu scripti tekrar çalıştırın")
        print("   3. PCAP dosya yolunu girin")

if __name__ == "__main__":
    main()
