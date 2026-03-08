"""
🖥️ STREAMLIT ARAYÜZ – DOSYA YÜKLE, SALDIRI VAR MI GÖR
=====================================================

Modlar:
- UNSW-NB15 tabanlı model:
    - Eğitim verisi: data/features/advanced_features.csv + data/labeled/labeled_traffic.csv
    - Model: RandomForestClassifier (class_weight='balanced_subsample')
    - Özellikler: 12 advanced feature

KULLANIM:
    streamlit run streamlit_app.py

Sonra arayüzden CSV yükleyebilirsin.
"""

import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FEATURES_FILE = DATA_DIR / "features" / "advanced_features.csv"
LABELED_FILE = DATA_DIR / "labeled" / "labeled_traffic.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODEL_COMPARISON_REPORT = OUTPUTS_DIR / "model_comparison_report.json"
CICIDS_SUPERVISED_REPORT = OUTPUTS_DIR / "cicids2017_supervised_report.json"
CICIDS_CROSS_REPORT = OUTPUTS_DIR / "cicids2017_cross_day_report.json"


@st.cache_resource
def load_unsw_model():
    """UNSW-NB15 advanced_features ile RandomForest modeli eğit."""
    if not FEATURES_FILE.exists() or not LABELED_FILE.exists():
        st.error(
            f"UNSW feature veya label dosyası bulunamadı.\n"
            f"Önce şu komutları çalıştır:\n"
            f"  python scripts/label_traffic.py\n"
            f"  python scripts/advanced_features.py"
        )
        st.stop()

    X = pd.read_csv(FEATURES_FILE)
    df_labels = pd.read_csv(LABELED_FILE)

    if "label" not in df_labels.columns:
        st.error("'label' kolonu labeled_traffic.csv içinde yok.")
        st.stop()

    y = (df_labels["label"] == "anomaly").astype(int)
    min_len = min(len(X), len(y))
    X = X.iloc[:min_len]
    y = y.iloc[:min_len]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_scaled, y)

    return model, scaler, list(X.columns)


def predict_on_uploaded_csv(model, scaler, feature_names, df: pd.DataFrame):
    """Yüklenen CSV üzerinde tahmin yap."""
    # Eğer label varsa ayır
    y_true = None
    label_col = None
    for cand in ["label", "Label", "y", "attack", "attack_cat"]:
        if cand in df.columns:
            label_col = cand
            break

    if label_col is not None:
        y_true = (df[label_col].astype(str).str.lower().isin(["1", "anomaly", "attack"])).astype(
            int
        )
        df = df.drop(columns=[label_col])

    # Sadece eğitimde kullanılan feature’ları al
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        st.warning(
            "Bazı beklenen feature'lar dosyada yok, bunlar 0 kabul edilecek:\n"
            + ", ".join(missing)
        )
    X = pd.DataFrame()
    for f in feature_names:
        if f in df.columns:
            X[f] = pd.to_numeric(df[f], errors="coerce")
        else:
            X[f] = 0.0

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_scaled = scaler.transform(X)

    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    return y_true, y_pred, y_proba


def page_unsw_rf_demo():
    """UNSW-NB15 tabanlı RandomForest demo sayfası."""
    st.markdown(
        """
### 🎯 UNSW‑NB15 – RandomForest Demo

Bu sayfada:
- `data/features/advanced_features.csv` ve `data/labeled/labeled_traffic.csv` ile eğitilmiş
  **RandomForest** modeli kullanılır.
- Kendi CSV dosyanı yükleyip içinde **saldırı (anomaly) var mı** görebilirsin.

> Not: Yüklediğin CSV'nin kolonları mümkün olduğunca `advanced_features.csv`'deki
> 12 feature'a (duration, orig_bytes, resp_bytes, ...) benzemeli.
"""
    )

    with st.spinner("UNSW‑NB15 modeli yükleniyor / eğitiliyor..."):
        model, scaler, feature_names = load_unsw_model()

    uploaded = st.file_uploader("CSV dosyası yükle", type=["csv"])

    if uploaded is None:
        st.info("İlk olarak test etmek istediğin trafiğin CSV dosyasını yükle.")
        return

    try:
        df_up = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"CSV okunamadı: {e}")
        return

    st.write("Yüklenen dosya boyutu:", df_up.shape)

    y_true, y_pred, y_proba = predict_on_uploaded_csv(model, scaler, feature_names, df_up)

    st.sidebar.subheader("⚙️ UNSW Ayarları")
    threshold = st.sidebar.slider(
        "Anomali eşiği (1'e yaklaştıkça model daha seçici olur)",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help=(
            "Eşiği düşürürsen daha çok bağlantı 'ANOMALY' diye işaretlenir (daha hassas), "
            "yükseltirsen daha az ama daha emin olunan anomali yakalanır."
        ),
    )

    # Olasılığa (y_proba) göre eşiği uygula
    y_pred = (y_proba >= threshold).astype(int)

    df_result = df_up.copy()
    df_result["anomaly_score"] = y_proba
    df_result["prediction"] = np.where(y_pred == 1, "ANOMALY", "NORMAL")

    # Özet metrikler
    total = len(df_result)
    anomalies = (y_pred == 1).sum()
    st.subheader("📊 Genel Özet")
    col1, col2, col3 = st.columns(3)
    col1.metric("Toplam kayıt", total)
    col2.metric("Tahmin edilen anomali sayısı", anomalies)
    col3.metric("Anomali oranı", f"{anomalies/total*100:.2f}%")

    if y_true is not None:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        st.subheader("🎯 Gerçek etiketler varsa – Performans")
        st.write(f"**Accuracy**: {acc:.4f}")
        st.write(f"**Precision**: {prec:.4f}")
        st.write(f"**Recall**: {rec:.4f}")
        st.write(f"**F1-Score**: {f1:.4f}")
        st.write("Confusion Matrix (satır: gerçek, sütun: tahmin)")
        st.table(
            pd.DataFrame(
                cm,
                index=["Gerçek NORMAL", "Gerçek ANOMALY"],
                columns=["Tahmin NORMAL", "Tahmin ANOMALY"],
            )
        )

    # En şüpheli N bağlantı
    st.subheader("🚨 En yüksek anomali skoruna sahip ilk 50 kayıt")
    st.dataframe(
        df_result.sort_values("anomaly_score", ascending=False).head(50),
        use_container_width=True,
    )


def page_custom_dataset_results():
    """Kullanıcının kendi PCAP verilerinden elde edilen sonuçlar."""
    import matplotlib.pyplot as plt
    
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
                    st.metric("F1-Score", f"{row['f1_score']*100:.2f}%")
                with col5:
                    st.metric("ROC-AUC", f"{row['roc_auc']*100:.2f}%")
                
                # Performans yorumu
                if row['f1_score'] > 0.95:
                    st.success("✅ **Mükemmel Performans!** Model çok iyi çalışıyor.")
                elif row['f1_score'] > 0.85:
                    st.info("👍 **İyi Performans.** Kabul edilebilir seviyede.")
                else:
                    st.warning("⚠️ **İyileştirme gerekebilir.**")
        
        # En iyi modeli belirt
        best_ae = ae_models.loc[ae_models['f1_score'].idxmax()]
        st.success(f"🏆 **En İyi Autoencoder:** {best_ae['Model']} (F1-Score: {best_ae['f1_score']:.4f})")
    
    # Tüm modellerin karşılaştırması
    st.markdown("---")
    st.markdown("### 📈 **Tüm Modellerin Karşılaştırması**")
    
    df_sorted = df.sort_values('f1_score', ascending=False)
    st.dataframe(df_sorted, use_container_width=True)
    
    # Tarih bilgisi
    comparison_date = report.get("comparison_date", "Bilinmiyor")
    st.caption(f"📅 Test tarihi: {comparison_date}")


def page_autoencoder_overview():
    """Projeye başlangıç noktası: Autoencoder tabanlı IDS özeti."""
    st.markdown(
        """
# 🧠 Projenin Kalbi: Autoencoder Tabanlı Anomali Tespiti

> **💡 Bu proje neden Autoencoder'a odaklanıyor?**
> 
> Klasik supervised modeller (RandomForest, SVM vb.) sadece **bilinen saldırıları** tespit edebilir.
> Ama **Autoencoder'lar farklı çalışır**: Sadece normal trafiği öğrenirler ve bundan sapan 
> **her şeyi anomali** olarak işaretlerler. Bu sayede **hiç görmediği saldırıları bile** yakalayabilir!

---

## 🎯 Projede Kullanılan Autoencoder Modelleri

### 1️⃣ **LSTM Autoencoder** ⭐ (Ana Model)
**Özel Yetenek:** Zaman serisi analizi ile IP bazlı davranış öğrenme

**Ne yapar?**
- Her IP adresinin **zaman içindeki davranış paternini** öğrenir
- Slow scan, low-and-slow gibi **yavaş saldırıları** yakalar
- Session hijacking gibi **davranış değişikliklerini** tespit eder

**Teknik Detaylar:**
- Sequence Length: 10 (10'lu zaman pencereleri)
- LSTM katmanları: [64, 32] units
- Bottleneck: 16 dimensional latent space

### 2️⃣ **Standard Autoencoder**
**Özel Yetenek:** Hızlı ve hafif anomali tespiti

**Ne yapar?**
- Her bağlantıyı **bağımsız olarak** analiz eder
- Anormal byte/packet oranlarını yakalar
- Hızlı inference için ideal

**Teknik Detaylar:**
- Encoding dim: 16
- Dense layers ile basit ama etkili
- Reconstruction error threshold ile tespit

---

## 📊 Model Karşılaştırması
"""
    )

    if MODEL_COMPARISON_REPORT.exists():
        try:
            with MODEL_COMPARISON_REPORT.open(encoding="utf-8") as f:
                report = json.load(f)
            model_results = report.get("model_results", {})
            if model_results:
                df = pd.DataFrame.from_dict(model_results, orient="index")
                df = df.reset_index().rename(columns={"index": "Model"})
                
                # Autoencoder modellerini öne çıkar
                ae_models = df[df["Model"].str.contains("Autoencoder", case=False)]
                other_models = df[~df["Model"].str.contains("Autoencoder", case=False)]
                
                if not ae_models.empty:
                    st.markdown("### 🌟 **Autoencoder Modelleri** (Projenin Çekirdeği)")
                    
                    # Autoencoder'ları vurgulu göster
                    st.dataframe(
                        ae_models.style.background_gradient(cmap='Greens', subset=['f1_score', 'roc_auc']),
                        use_container_width=True
                    )
                    
                    # En iyi Autoencoder'ı belirt
                    best_ae = ae_models.loc[ae_models['f1_score'].idxmax()]
                    st.success(f"🏆 **En İyi Autoencoder:** {best_ae['Model']} (F1-Score: {best_ae['f1_score']:.4f})")
                
                if not other_models.empty:
                    st.markdown("### 📋 Diğer Modeller (Karşılaştırma için)")
                    st.dataframe(other_models, use_container_width=True)
                
                # Tüm modelleri birlikte göster
                st.markdown("### 📈 Tüm Modellerin Karşılaştırması")
                st.dataframe(df, use_container_width=True)
                
        except Exception as e:
            st.warning(f"Model karşılaştırma raporu okunamadı: {e}")
    else:
        st.info(
            """
**📝 Model karşılaştırma raporu henüz yok.**

Autoencoder ve diğer modellerin performansını görmek için:

```bash
python scripts/experiements/compare_all_models.py
```

Bu komut:
- ✅ LSTM Autoencoder ve Standard Autoencoder'ı eğitir
- ✅ Isolation Forest, One-Class SVM gibi diğer modelleri test eder
- ✅ Sonuçları `outputs/model_comparison_report.json` olarak kaydeder
- ✅ Performans grafiklerini oluşturur
"""
        )

    st.markdown(
        """
---

## 🔬 Neden Autoencoder Supervised Modellerden Daha İyi?

| Özellik | Supervised (RF, SVM) | Autoencoder |
|---------|---------------------|-------------|
| **Eğitim Verisi** | Hem normal hem saldırı gerekli | Sadece normal trafik yeterli ✅ |
| **Yeni Saldırılar** | Görmediği saldırıları kaçırır ❌ | Hiç görmediği saldırıları yakalar ✅ |
| **Veri İhtiyacı** | Çok fazla etiketli veri gerekli | Az veri ile çalışır ✅ |
| **Adaptasyon** | Yeni saldırı için yeniden eğitim | Otomatik adapte olur ✅ |

---

## 🚀 Hızlı Başlangıç

Autoencoder modellerini test etmek için:

```bash
# 1. Veriyi hazırla
python scripts/merge_and_label_logs.py

# 2. Feature'ları çıkar
python scripts/data_pipeline/advanced_features.py

# 3. Tüm modelleri karşılaştır (Autoencoder dahil)
python scripts/experiements/compare_all_models.py

# 4. Streamlit arayüzünü aç
streamlit run streamlit_app.py
```
"""
    )



def page_cicids_results():
    """CICIDS2017 supervised ve cross‑day sonuçlarının özeti."""
    st.subheader("📚 CICIDS2017 Sonuç Özeti")
    st.markdown(
        """
Bu bölümde CICIDS2017 veri seti üzerinde yaptığın **supervised** ve
**cross‑day (Tuesday → Wednesday)** deneylerin özetini gösteriyoruz.

Amaç:
- Aynı gün içinde supervised modellerin ne kadar iyi olduğunu,
- Gün değiştiğinde (dağılım kayınca) performansın nasıl bozulduğunu
  net bir şekilde görmek ve **autoencoder yaklaşımının neden önemli olduğunu** vurgulamak.
"""
    )

    cols = st.columns(2)

    # Supervised rapor
    if CICIDS_SUPERVISED_REPORT.exists():
        try:
            with CICIDS_SUPERVISED_REPORT.open(encoding="utf-8") as f:
                sup = json.load(f)
            with cols[0]:
                st.markdown("### ✅ Aynı Gün (Tuesday, train/test split)")
                info = sup.get("data_info", {})
                st.write(
                    f"Toplam örnek: **{info.get('total_samples', '?')}**, "
                    f"Benign: **{info.get('benign_samples', '?')}**, "
                    f"Saldırı: **{info.get('attack_samples', '?')}**"
                )
                res = pd.DataFrame.from_dict(sup.get("results", {}), orient="index")
                res = res.reset_index().rename(columns={"index": "Model"})
                st.table(res[["Model", "accuracy", "precision", "recall", "f1", "roc_auc"]])
        except Exception as e:
            st.warning(f"Supervised rapor okunamadı: {e}")
    else:
        cols[0].info(
            "Supervised CICIDS raporu bulunamadı. "
            "`python scripts/model_supervised/cicids2017_supervised.py` komutunu çalıştır."
        )

    # Cross‑day rapor
    if CICIDS_CROSS_REPORT.exists():
        try:
            with CICIDS_CROSS_REPORT.open(encoding="utf-8") as f:
                cross = json.load(f)
            with cols[1]:
                st.markdown("### ⚠️ Cross‑Day (Train: Tuesday → Test: Wednesday)")
                tinfo = cross.get("train_info", {})
                teinfo = cross.get("test_info", {})
                st.write(
                    f"Train Tuesday: **{tinfo.get('total_samples', '?')}** "
                    f"(Benign: {tinfo.get('benign_samples', '?')}, "
                    f"Saldırı: {tinfo.get('attack_samples', '?')})"
                )
                st.write(
                    f"Test Wednesday: **{teinfo.get('total_samples', '?')}** "
                    f"(Benign: {teinfo.get('benign_samples', '?')}, "
                    f"Saldırı: {teinfo.get('attack_samples', '?')})"
                )
                res = pd.DataFrame.from_dict(cross.get("results", {}), orient="index")
                res = res.reset_index().rename(columns={"index": "Model"})
                st.table(res[["Model", "accuracy", "precision", "recall", "f1", "roc_auc"]])

                st.markdown(
                    """
Gördüğün gibi:
- Aynı gün içinde metrikler çok yüksekken,
- Gün değişince özellikle **recall** ve **F1** ciddi şekilde düşüyor.

Bu da **sadece supervised modele güvenmek yerine**, autoencoder ve zaman serisi
yaklaşımını merkeze almanın neden önemli olduğunu gösteriyor.
"""
                )
        except Exception as e:
            st.warning(f"Cross‑day rapor okunamadı: {e}")
    else:
        cols[1].info(
            "Cross‑day CICIDS raporu bulunamadı. "
            "`python scripts/experiements/cicids2017_cross_day.py` komutunu çalıştır."
        )


def main():
    st.set_page_config(page_title="Network Anomaly Detection", layout="wide")
    st.title("🔒 Network Anomaly Detection System")

    menu = [
        "🎯 SENİN VERİ SETİN - Test Sonuçları",  # YENİ - Kullanıcının verisi
        "🔍 Kendi PCAP'ini Analiz Et",  # YENİ - Custom PCAP upload
        "⚙️ Model Stratejisi Seç",  # YENİ - Ensemble selection
        "🧠 Autoencoder Proje Özeti",
        "📊 CICIDS2017 Supervised Sonuçları",
        "🔄 CICIDS2017 Cross-Day Test",
        "📈 Model Karşılaştırma",
    ]
    choice = st.sidebar.selectbox("Menü", menu)

    if choice == "🎯 SENİN VERİ SETİN - Test Sonuçları":
        page_custom_dataset_results()
    elif choice == "🔍 Kendi PCAP'ini Analiz Et":
        # Import custom PCAP analysis page
        import sys
        sys.path.insert(0, str(Path.cwd()))
        from streamlit_analyze_custom import page_analyze_custom_pcap
        page_analyze_custom_pcap()
    elif choice == "⚙️ Model Stratejisi Seç":
        # Import ensemble selector page
        import sys
        sys.path.insert(0, str(Path.cwd()))
        from streamlit_ensemble_selector import page_ensemble_selector
        page_ensemble_selector()
    elif choice == "🧠 Autoencoder Proje Özeti":
        page_autoencoder_overview()
    elif choice == "📊 CICIDS2017 Supervised Sonuçları":
        # Assuming page_cicids_results() handles both supervised and cross-day,
        # or a new function for supervised results is needed.
        # For now, mapping to the existing page_cicids_results which has both.
        page_cicids_results()
    elif choice == "🔄 CICIDS2017 Cross-Day Test":
        page_cicids_results() # Re-using the existing page for now
    elif choice == "📈 Model Karşılaştırma":
        page_autoencoder_overview() # Re-using this page as it contains model comparison
    else:
        # Fallback, though with selectbox, this should ideally not be reached
        st.info("Lütfen bir bölüm seçin.")


if __name__ == "__main__":
    main()


