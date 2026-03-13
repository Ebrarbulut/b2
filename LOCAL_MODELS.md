# Kendi bilgisayarında tüm modelleri ekleme

Backend ve PWA'da **Standard Autoencoder** ile **Isolation Forest** (ve isteğe bağlı ensemble) kullanmak için modelleri kendi bilgisayarında üretmen gerekir. Model dosyaları repo'da tutulmaz (.gitignore); her ortamda bir kez bu adımlar çalıştırılır.

## Gereksinimler

- `data/features/advanced_features.csv` mevcut olmalı.
- `data/labeled/labeled_traffic.csv` mevcut olmalı.

(Bunları offline pipeline ile üretmiş olmalısın: `merge_and_label_logs.py`, `data_pipeline/advanced_features.py` vb.)

## Adımlar

1. **Sanal ortamı aç, proje kökünde ol:**

   ```bash
   cd "C:\Users\EBRAR BULUT\OneDrive\Masaüstü\B2"
   .\venv\Scripts\activate
   ```

2. **Tüm çekirdek modelleri eğitip kaydet (Standard AE + Isolation Forest):**

   ```bash
   python scripts/experiements/compare_core_models.py
   ```

   Bu script:
   - Veriyi yükler, train/test böler.
   - **Standard Autoencoder** eğitir; `models/standard_autoencoder.keras`, `models/standard_ae_scaler.pkl`, kökte `standard_ae_config.pkl` üretir.
   - **Isolation Forest** eğitir; `models/isolation_forest.pkl`, `models/if_scaler.pkl`, kökte `if_config.pkl` üretir.
   - Sonuçları `outputs/core_model_comparison.csv` dosyasına yazar.

3. **Kontrol:**

   - `models/` içinde şunlar olsun: `standard_autoencoder.keras`, `standard_ae_scaler.pkl`, `isolation_forest.pkl`, `if_scaler.pkl`
   - Proje kökünde: `standard_ae_config.pkl`, `if_config.pkl`

4. **Backend’i çalıştır:**

   ```bash
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

   Tarayıcıda `http://localhost:8000/api/models` açınca her iki modelin de listelendiğini görebilirsin.

5. **Frontend’i çalıştır (ayrı terminal):**

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

   `frontend/.env` içinde `VITE_API_URL=http://localhost:8000` olsun. Sonra `http://localhost:5173` aç; "Model karşılaştırma" kartında her iki model "Yüklü" görünür.

## Not

- Render’da canlı backend’te model dosyaları yok; orada sadece **demo** mod çalışır. Canlıda gerçek modelleri kullanmak istersen Render’a özel bir build adımıyla bu dosyaları üretmen veya harici depodan alman gerekir (bu dokümanda anlatılmaz).
