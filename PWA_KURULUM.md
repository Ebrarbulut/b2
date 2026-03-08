# PWA'ya Geçiş – Senin Yapacakların (Adım Adım)

Bu dosyada **sadece senin yapman gerekenler** sırayla ve kısaca anlatılıyor. Backend ve frontend iskeleti projeye eklendi; aşağıdaki adımlarla ortamı kurup çalıştıracaksın.

---

## Ön koşul

- **Node.js** yüklü olmalı (v18 veya üzeri önerilir).  
  Kontrol: Terminalde `node -v` yaz. Versiyon görünmüyorsa [nodejs.org](https://nodejs.org) adresinden indirip kur.

---

## Adım 1: Backend’i çalıştır

Backend, tarayıcıdaki PWA’nın istek atacağı API (FastAPI) sunucusudur.

1. **Terminali aç** ve proje kök klasörüne geç:
   ```text
   C:\Users\EBRAR BULUT\OneDrive\Masaüstü\B2
   ```

2. **Sanal ortamı etkinleştir** (zaten varsa):
   ```bash
   venv\Scripts\activate
   ```

3. **Backend bağımlılıklarını yükle** (ilk seferde bir kere):
   ```bash
   pip install fastapi uvicorn
   ```
   (Proje kökündeki `requirements.txt` zaten TensorFlow vb. içeriyor; sadece API için FastAPI + uvicorn yeterli.)

4. **Backend’i başlat**:
   ```bash
   uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
   ```

5. **Kontrol:** Tarayıcıda şu adresi aç:  
   `http://127.0.0.1:8000/health`  
   Sayfada `{"status":"ok", ...}` benzeri bir JSON görmelisin. Görüyorsan backend ayakta demektir.

6. Bu terminali **açık bırak** (kapatırsan API çalışmaz).

---

## Adım 2: Frontend bağımlılıklarını yükle

Frontend, PWA arayüzünün (React + Vite) kaynak kodu. Önce bağımlılıkları kurman gerekiyor.

1. **Yeni bir terminal** aç (backend’in çalıştığı terminali kapatma).

2. Yine proje kökünde olduğundan emin ol:
   ```text
   C:\Users\EBRAR BULUT\OneDrive\Masaüstü\B2
   ```

3. **Frontend klasörüne gir**:
   ```bash
   cd frontend
   ```

4. **npm ile paketleri kur** (ilk seferde bir kere, birkaç dakika sürebilir):
   ```bash
   npm install
   ```

   Hata alırsan: `npm install --legacy-peer-deps` dene.

---

## Adım 3: Frontend’i (PWA) geliştirme modunda çalıştır

1. Hâlâ **frontend** klasöründe olduğunu kontrol et (`cd frontend` yaptıysan burada olmalısın).

2. **Geliştirme sunucusunu başlat**:
   ```bash
   npm run dev
   ```

3. Terminalde şuna benzer bir satır çıkar:
   ```text
   Local:   http://localhost:5173/
   ```

4. **Tarayıcıda** şu adresi aç:  
   `http://localhost:5173`  

5. Açılan sayfada:
   - **“Health kontrol”** butonuna tıkla → Backend’e istek gider; altta `{"status":"ok", ...}` görünür.
   - **“Skor test”** butonuna tıkla → API’den anomali skoru döner; skorlar sayfada yazılır.

   Böylece hem frontend hem backend’in birlikte çalıştığını doğrularsın.

6. Bu terminali de **açık bırak**.

---

## Adım 4: PWA’yı “uygulama” gibi kur (isteğe bağlı)

1. `http://localhost:5173` açıkken **Chrome veya Edge** kullan.

2. Adres çubuğunun sağında **“Uygulamayı yükle” / “Install”** ikonuna tıkla (veya menü → “Uygulamayı yükle”).

3. Onaylayınca masaüstünde / başlat menüsünde “Network Anomaly Detection” kısayolu oluşur; tıklayınca uygulama penceresi açılır (tarayıcı çerçevesi olmadan).

---

## Özet – Hangi komutları nerede çalıştırıyorsun?

| Sıra | Nerede (klasör) | Komut | Açıklama |
|------|------------------|--------|----------|
| 1 | Proje kökü `B2` | `venv\Scripts\activate` | Sanal ortamı aç |
| 2 | Proje kökü `B2` | `pip install fastapi uvicorn` | Backend bağımlılıkları (ilk sefer) |
| 3 | Proje kökü `B2` | `uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000` | API’yi başlat, terminali açık bırak |
| 4 | Proje kökü `B2` | `cd frontend` | Frontend klasörüne geç |
| 5 | `frontend` | `npm install` | Frontend bağımlılıkları (ilk sefer) |
| 6 | `frontend` | `npm run dev` | PWA arayüzünü başlat, tarayıcıda 5173 aç |

---

## Sık karşılaşılan durumlar

- **“Health kontrol” çalışmıyor / hata veriyor**  
  Backend’in 8000 portunda çalıştığından emin ol. Aynı terminalde `uvicorn backend.main:app ...` komutunu tekrar çalıştır.

- **“Skor test” demo skor döndürüyor**  
  `data/features/advanced_features.csv` yoksa veya model yüklenemezse API “demo” modunda rastgele skor verir. Önce `scripts/merge_and_label_logs.py` ve `scripts/data_pipeline/advanced_features.py` çalıştırıp veriyi üret; sonra backend’i yeniden başlat.

- **npm install hata veriyor**  
  `npm install --legacy-peer-deps` dene veya Node sürümünü 18+ yap.

Bu adımları tamamladığında proje PWA tarafında çalışır durumda olur; sonrasında arayüzü genişletmek (dosya yükleme, grafikler vb.) için istersen devam edebiliriz.
