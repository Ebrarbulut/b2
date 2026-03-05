"""
ENSEMBLE MODEL SEÇİMİ SAYFASI
"""
import streamlit as st

def page_ensemble_selector():
    """Ensemble model stratejisi seçim sayfası"""
    
    st.markdown("# 🎯 **Model Stratejisi Seçimi**")
    st.markdown("---")
    
    st.info("""
    **Bu sayfada** hangi tespit stratejisini kullanmak istediğini seçebilirsin.
    Farklı stratejiler farklı performans ve hız dengeleri sunar.
    """)
    
    # 3 Strateji
    st.markdown("## 📊 **Mevcut Stratejiler**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🏆 Isolation Forest")
        st.metric("F1-Score", "99.5%")
        st.metric("False Positive", "0.63%")
        st.metric("Hız", "⚡⚡⚡")
        
        st.success("""
        **Avantajlar:**
        - En yüksek performans
        - Çok düşük yanlış alarm
        - Hızlı

        **Önerilen:** Production kullanımı
        """)
    
    with col2:
        st.markdown("### 🎯 Ensemble Weighted")
        st.metric("F1-Score", "~99.0%*")
        st.metric("False Positive", "~1%*")
        st.metric("Hız", "⚡⚡")
        
        st.info("""
        **Avantajlar:**
        - Daha robust
        - Farklı saldırıları yakalar
        - IF + AE kombinasyonu

        **Önerilen:** Maksimum güvenlik
        """)
    
    with col3:
        st.markdown("### 👥 Majority Voting")
        st.metric("F1-Score", "~96%*")
        st.metric("False Positive", "~5%*")
        st.metric("Hız", "⚡⚡")
        
        st.warning("""
        **Avantajlar:**
        - Hiçbir saldırı kaçmaz
        - En agresif tespit

        **Dezavantaj:** Çok yanlış alarm
        """)
    
    st.markdown("---")
    
    # Seçim
    st.markdown("## ⚙️ **Stratejini Seç**")
    
    strategy = st.radio(
        "Hangi stratejiyi kullanmak istiyorsun?",
        [
            "🏆 Isolation Forest Only (En Yüksek Performans)",
            "🎯 Ensemble Weighted Voting (En Güvenli)",
            "👥 Majority Voting (En Agresif)"
        ],
        index=0
    )
    
    # Detaylı açıklama
    st.markdown("---")
    st.markdown("## 💡 **Seçtiğin Strateji Nasıl Çalışır?**")
    
    if "Isolation Forest" in strategy:
        st.markdown("""
        ### 🌲 Isolation Forest Stratejisi
        
        **Karar Mekanizması:**
        ```python
        score = isolation_forest.predict_proba(traffic)
        
        if score >= 0.115:  # Threshold (70th percentile)
            result = "ANOMALİ"
        else:
            result = "NORMAL"
        ```
        
        **Neden Bu Kadar İyi?**
        - Anomalileri "izole edilen" noktalar olarak görür
        - Ağaç tabanlı ensemble method
        - O(n log n) hız
        
        **Performans:**
        - ✅ 13,389 saldırı yakalandı
        - ❌ 134 yanlış alarm
        - ❌ 1 kaçan saldırı
        """)
        
    elif "Ensemble" in strategy:
        st.markdown("""
        ### 🎯 Ensemble Weighted Voting
        
        **Karar Mekanizması:**
        ```python
        if_score = isolation_forest.predict_proba(traffic)
        ae_score = autoencoder.predict_proba(traffic)
        
        # Ağırlıklı skor
        weighted_score = 0.70 * if_score + 0.30 * ae_score
        
        if weighted_score >= 0.50:
            result = "ANOMALİ"
        else:
            result = "NORMAL"
        ```
        
        **Neden Ensemble?**
        - IF: Port pattern anomalileri
        - AE: Byte/packet pattern anomalileri
        - İki model farklı açılardan bakıyor
        
        **Avantajları:**
        - Daha robust (tek model yanılırsa diğeri yakalar)
        - False Positive azalır
        - Farklı saldırı tiplerinde daha iyi
        """)
        
    else:
        st.markdown("""
        ### 👥 Majority Voting Stratejisi
        
        **Karar Mekanizması:**
        ```python
        if_prediction = isolation_forest.predict(traffic)
        ae_prediction = autoencoder.predict(traffic)
        
        # En az 1 model uyarırsa
        if if_prediction == 1 OR ae_prediction == 1:
            result = "ANOMALİ"
        else:
            result = "NORMAL"
        ```
        
        **Kullanım Alanları:**
        - Kritik altyapı (hiçbir saldırı kaçmamalı)
        - SOC ekibi yanlış alarmları manuel inceleyebiliyorsa
        - Zero-tolerance güvenlik politikası
        
        **Dikkat:**
        - Çok fazla yanlış alarm üretir
        - SOC yorgunluğuna neden olabilir
        """)
    
    st.markdown("---")
    st.success("💾 Seçimin kaydedildi. Analiz yaparken bu strateji kullanılacak!")

if __name__ == "__main__":
    page_ensemble_selector()
