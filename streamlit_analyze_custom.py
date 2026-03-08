"""
KULLANICI PCAP ANALİZ ARACI
Kullanıcıların kendi PCAP dosyalarını yükleyip analiz etmelerini sağlar
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path.cwd()
sys.path.insert(0, str(BASE_DIR / "scripts"))

from models_unsupervised.isolation_forest_detector import IsolationForestDetector

def page_analyze_custom_pcap():
    """Kullanıcının kendi PCAP'ini analiz etme sayfası - GELİŞTİRİLMİŞ"""
    
    st.markdown("# 🔍 **Kendi PCAP Dosyanı Analiz Et**")
    st.markdown("---")
    
    # Kullanım talimatları
    with st.expander("📖 Nasıl Kullanılır?", expanded=True):
        st.markdown("""
        ### Adımlar:
        1. **PCAP Dosyası Yükle**: `.pcap` formatında ağ trafiği dosyanı seç
        2. **Zeek ile İşle**: PCAP'i conn.log formatına çevir (otomatik)
        3. **Özellik Çıkarımı**: 12 özellik otomatik hesaplanır
        4. **Anomali Tespiti**: Eğitilmiş Isolation Forest modeli ile analiz
        5. **Sonuçları Gör**: Anomali ve normal trafiği ayrıştır, indir
        
        ### Desteklenen Dosyalar:
        - ✅ `.pcap` - Wireshark PCAP dosyası
        - ✅ `.log` - Zeek conn.log dosyası  
        - ✅ `.csv` - Özellikler çıkarılmış CSV
        
        ### Not:
        - PCAP işleme 5-10 dakika sürebilir (dosya boyutuna bağlı)
        - Maksimum dosya boyutu: 200MB
        """)
    
    st.markdown("## 📁 **Adım 1: Dosya Yükle**")
    
    upload_type = st.radio(
        "Hangi dosya tipini yükleyeceksin?",
        ["PCAP Dosyası (.pcap)", "Zeek conn.log", "CSV (özellikler çıkarılmış)"],
        help="PCAP'i Zeek ile işleyip conn.log'a çevirebiliriz"
    )
    
    uploaded_file = st.file_uploader(
        "Dosyayı seç",
        type=['pcap', 'log', 'csv']
    )
    
    if uploaded_file is not None:
        st.success(f"✅ Dosya yüklendi: {uploaded_file.name}")
        
        # Dosyayı kaydet
        temp_path = Path("temp") / uploaded_file.name
        temp_path.parent.mkdir(exist_ok=True)
        
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # PCAP ise conn.log'a çevir
        if upload_type == "PCAP Dosyası (.pcap)":
            st.markdown("## 🔄 **Adım 2: PCAP'i İşle**")
            
            if st.button("🚀 PCAP'i conn.log'a Çevir"):
                with st.spinner("Zeek ile işleniyor..."):
                    try:
                        # PCAP to conn.log conversion
                        import subprocess
                        result = subprocess.run(
                            ['python', 'scripts/pcap_to_connlog_simple.py', str(temp_path)],
                            capture_output=True,
                            text=True
                        )
                        
                        if result.returncode == 0:
                            st.success("✅ conn.log oluşturuldu!")
                            conn_log_path = temp_path.with_suffix('.log')
                        else:
                            st.error(f"❌ Hata: {result.stderr}")
                            return
                    except Exception as e:
                        st.error(f"❌ İşlem hatası: {e}")
                        return
        
        # conn.log ise feature extraction
        elif upload_type == "Zeek conn.log":
            st.markdown("## 🔄 **Adım 2: Özellik Çıkarımı**")
            conn_log_path = temp_path
            
            if st.button("🚀 Özellikleri Çıkar"):
                with st.spinner("Özellikler çıkarılıyor..."):
                    try:
                        # Feature extraction (basitleştirilmiş)
                        df = pd.read_csv(conn_log_path, sep='\t', comment='#')
                        
                        # Basit özellikler
                        features = pd.DataFrame({
                            'duration': df.get('duration', 0),
                            'orig_bytes': df.get('orig_bytes', 0),
                            'resp_bytes': df.get('resp_bytes', 0),
                            'orig_pkts': df.get('orig_pkts', 0),
                            'resp_pkts': df.get('resp_pkts', 0),
                        })
                        
                        features = features.fillna(0)
                        st.success(f"✅ {len(features)} bağlantı işlendi!")
                        
                    except Exception as e:
                        st.error(f"❌ Özellik çıkarımı hatası: {e}")
                        return
        
        # CSV ise direkt yükle
        else:
            st.markdown("## 📊 **Adım 2: Veri Yüklendi**")
            features = pd.read_csv(temp_path)
            st.success(f"✅ {len(features)} örnek yüklendi!")
        
        # Adım 3: Model ile analiz
        st.markdown("## 🤖 **Adım 3: Anomali Tespiti**")
        
        if st.button("🔍 Analiz Et"):
            with st.spinner("Model ile analiz ediliyor..."):
                try:
                    # Model yükle
                    model_path = BASE_DIR / "models" / "isolation_forest.pkl"
                    
                    if not model_path.exists():
                        st.warning("⚠️ Model bulunamadı. Yeni model eğitiliyor...")
                        # Basit model eğit
                        detector = IsolationForestDetector(contamination=0.1)
                        X_train = pd.read_csv(BASE_DIR / "data/features/advanced_features.csv")
                        detector.train(X_train[:1000])  # İlk 1000 örnek
                    else:
                        detector = IsolationForestDetector.load_model(str(model_path))
                    
                    # Tahmin yap
                    predictions, scores = detector.predict(features)
                    
                    # Sonuçları göster
                    st.markdown("### 📊 **Analiz Sonuçları**")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Toplam Bağlantı", len(features))
                    with col2:
                        st.metric("Normal", (predictions == 0).sum())
                    with col3:
                        st.metric("⚠️ Anomali", (predictions == 1).sum())
                    
                    # Anomalileri listele
                    if (predictions == 1).sum() > 0:
                        st.markdown("### 🚨 **Tespit Edilen Anomaliler**")
                        
                        anomaly_df = features[predictions == 1].copy()
                        anomaly_df['anomaly_score'] = scores[predictions == 1]
                        anomaly_df = anomaly_df.sort_values('anomaly_score', ascending=False)
                        
                        st.dataframe(anomaly_df.head(50), use_container_width=True)
                        
                        # İndir butonu
                        csv = anomaly_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Anomalileri İndir (CSV)",
                            data=csv,
                            file_name="anomalies.csv",
                            mime="text/csv"
                        )
                    else:
                        st.success("✅ Anomali tespit edilmedi!")
                
                except Exception as e:
                    st.error(f"❌ Analiz hatası: {e}")
                    st.exception(e)
    
    else:
        st.info("👆 Yukarıdan bir dosya yükle")


# Test için
if __name__ == "__main__":
    page_analyze_custom_pcap()
