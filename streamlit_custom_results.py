def page_custom_dataset_results():
    """Kullanıcının kendi PCAP verilerinden elde edilen sonuçlar."""
    st.markdown(
        """
# 🎯 **SENİN VERİ SETİNDEN ELDE EDİLEN SONUÇLAR**

Bu sayfa, **senin PCAP dosyalarından** (`data/pcap/logs_*`) elde edilen test sonuçlarını gösteriyor.

---
"""
    )
    
    # Model comparison report'u kontrol et
    if not MODEL_COMPARISON_REPORT.exists():
        st.error(
            """
❌ **Model karşılaştırma raporu bulunamadı!**

Senin veri setini test etmek için şu adımları tamamla:

```bash
# 1. Logları birleştir ve etiketle
python scripts/merge_and_label_logs.py

# 2. Feature'ları çıkar
python scripts/data_pipeline/advanced_features.py

# 3. Modelleri eğit ve test et
python scripts/experiements/compare_all_models.py
```
"""
        )
        return
    
    # Raporu yükle
    try:
        with MODEL_COMPARISON_REPORT.open(encoding="utf-8") as f:
            report = json.load(f)
    except Exception as e:
        st.error(f"Rapor okunamadı: {e}")
        return
    
    # Veri bilgilerini göster
    st.markdown("## 📊 **Veri Seti Bilgileri**")
    data_info = report.get("data_info", {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Toplam Örnek", f"{data_info.get('total_samples', 0):,}")
    with col2:
        st.metric("Normal Trafik", f"{data_info.get('normal_samples', 0):,}")
    with col3:
        st.metric("Saldırı Trafiği", f"{data_info.get('anomaly_samples', 0):,}")
    with col4:
        st.metric("Özellik Sayısı", data_info.get('features', 0))
    
    # Veri dengesizliği uyarısı
    normal = data_info.get('normal_samples', 0)
    anomaly = data_info.get('anomaly_samples', 0)
    if normal > 0 and anomaly > 0:
        ratio = anomaly / normal
        if ratio > 10:
            st.warning(
                f"""
⚠️ **Veri Dengesizliği Tespit Edildi!**

- Normal/Anomaly oranı: **1:{ratio:.1f}**
- Bu dengesizlik, modelin performansını yanıltıcı gösterebilir
- Özellikle normal trafiği tespit etme yeteneği belirsiz
- Daha dengeli bir test için normal trafik örneklerini artırmayı düşün
"""
            )
    
    st.markdown("---")
    
    # Model sonuçlarını göster
    st.markdown("## 🏆 **Model Performans Sonuçları**")
    
    model_results = report.get("model_results", {})
    if not model_results:
        st.warning("Model sonuçları bulunamadı.")
        return
    
    # DataFrame oluştur
    df = pd.DataFrame.from_dict(model_results, orient="index")
    df = df.reset_index().rename(columns={"index": "Model"})
    
    # Autoencoder modellerini ayır
    ae_models = df[df["Model"].str.contains("Autoencoder", case=False)]
    other_models = df[~df["Model"].str.contains("Autoencoder", case=False)]
    
    # Autoencoder sonuçlarını vurgula
    if not ae_models.empty:
        st.markdown("### 🌟 **Autoencoder Modelleri** (Projenin Çekirdeği)")
        
        # Metrikler
        for idx, row in ae_models.iterrows():
            with st.expander(f"**{row['Model']}** - Detaylı Sonuçlar", expanded=True):
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Accuracy", f"{row['accuracy']*100:.2f}%")
                with col2:
                    st.metric("Precision", f"{row['precision']*100:.2f}%")
                with col3:
                    st.metric("Recall", f"{row['recall']*100:.2f}%")
                with col4:
                    st.metric("F1-Score", f"{row['f1_score']*100:.2f}%", 
                             delta=f"{(row['f1_score']-0.5)*100:.1f}%" if row['f1_score'] > 0.5 else None)
                with col5:
                    st.metric("ROC-AUC", f"{row['roc_auc']*100:.2f}%")
                
                # Performans yorumu
                if row['f1_score'] > 0.95:
                    st.success("✅ **Mükemmel Performans!** Model çok iyi çalışıyor.")
                elif row['f1_score'] > 0.85:
                    st.info("👍 **İyi Performans.** Kabul edilebilir seviyede.")
                elif row['f1_score'] > 0.70:
                    st.warning("⚠️ **Orta Performans.** İyileştirme gerekebilir.")
                else:
                    st.error("❌ **Zayıf Performans.** Model bu veri setinde başarısız.")
        
        # Tablo gösterimi
        st.markdown("#### 📋 Autoencoder Karşılaştırma Tablosu")
        st.dataframe(
            ae_models.style.background_gradient(cmap='Greens', subset=['f1_score', 'roc_auc']),
            use_container_width=True
        )
        
        # En iyi modeli belirt
        best_ae = ae_models.loc[ae_models['f1_score'].idxmax()]
        st.success(f"🏆 **En İyi Autoencoder:** {best_ae['Model']} (F1-Score: {best_ae['f1_score']:.4f})")
    
    # Diğer modeller
    if not other_models.empty:
        st.markdown("---")
        st.markdown("### 📊 **Diğer Modeller** (Karşılaştırma İçin)")
        st.dataframe(other_models, use_container_width=True)
    
    # Tüm modellerin karşılaştırması
    st.markdown("---")
    st.markdown("### 📈 **Tüm Modellerin Karşılaştırması**")
    
    # Sıralı tablo
    df_sorted = df.sort_values('f1_score', ascending=False)
    st.dataframe(
        df_sorted.style.background_gradient(cmap='RdYlGn', subset=['f1_score']),
        use_container_width=True
    )
    
    # Grafik gösterimi
    st.markdown("### 📊 **Görsel Karşılaştırma**")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # F1-Score karşılaştırması
    axes[0].barh(df_sorted['Model'], df_sorted['f1_score'], color='skyblue')
    axes[0].set_xlabel('F1-Score')
    axes[0].set_title('Model F1-Score Karşılaştırması')
    axes[0].set_xlim(0, 1)
    for i, v in enumerate(df_sorted['f1_score']):
        axes[0].text(v + 0.02, i, f'{v:.3f}', va='center')
    
    # ROC-AUC karşılaştırması
    axes[1].barh(df_sorted['Model'], df_sorted['roc_auc'], color='lightcoral')
    axes[1].set_xlabel('ROC-AUC')
    axes[1].set_title('Model ROC-AUC Karşılaştırması')
    axes[1].set_xlim(0, 1)
    for i, v in enumerate(df_sorted['roc_auc']):
        axes[1].text(v + 0.02, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Sonuç özeti
    st.markdown("---")
    st.markdown("## 💡 **Sonuç ve Öneriler**")
    
    best_model = df_sorted.iloc[0]
    st.info(
        f"""
**En İyi Model:** {best_model['Model']}
- F1-Score: {best_model['f1_score']:.4f}
- Accuracy: {best_model['accuracy']:.4f}

**Öneriler:**
1. Eğer normal trafik örnekleri azsa, daha fazla normal trafik toplayarak dengeyi iyileştir
2. Farklı threshold değerleri deneyerek precision/recall dengesini optimize et
3. Cross-validation ile modelin genelleme yeteneğini test et
4. Gerçek zamanlı tespit için en iyi modeli deploy et
"""
    )
    
    # Tarih bilgisi
    comparison_date = report.get("comparison_date", "Bilinmiyor")
    st.caption(f"📅 Test tarihi: {comparison_date}")


