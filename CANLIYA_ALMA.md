# Projeyi Canlıya Alma (Deploy)

Frontend (PWA) ve backend’i internette yayınlamak için aşağıdaki adımları sırayla uygula.

---

## Genel mantık

- **Frontend (React PWA)** → **Vercel** (ücretsiz, GitHub’dan otomatik deploy).
- **Backend (FastAPI)** → **Render** (ücretsiz tier; Python uygulaması çalıştırır).

Önce backend’i canlıya alıyorsun; sonra frontend’e canlı API adresini verip frontend’i deploy ediyorsun.

---

## Bölüm 1: Backend’i Render’da canlıya al

1. **GitHub’da projen güncel olsun**  
   Tüm değişiklikleri commit edip `main` (veya kullandığın branch) branch’ine push et.

2. **Render’a gir**  
   [render.com](https://render.com) → Sign up / Login (GitHub ile giriş yapabilirsin).

3. **Yeni Web Service**  
   Dashboard’da **New +** → **Web Service**.

4. **Repoyu bağla**  
   GitHub hesabını bağla, bu projenin repoyu seç (örn. `b2` veya `network_anomaly_detection_system`).

5. **Ayarlar**  
   - **Name:** `nad-api` (veya istediğin isim).  
   - **Region:** Frankfurt veya Oregon.  
   - **Branch:** `main`.  
   - **Root Directory:** Boş bırak (proje kökü).  
   - **Runtime:** Python 3.  
   - **Build Command:**  
     ```bash
     pip install -r requirements.txt && pip install fastapi uvicorn
     ```  
   - **Start Command:**  
     ```bash
     uvicorn backend.main:app --host 0.0.0.0 --port $PORT
     ```  
   - **Instance type:** Free.

6. **Environment variables (opsiyonel)**  
   - **CORS_ORIGINS:** İlk etapta boş bırakabilirsin; frontend’i deploy ettikten sonra Vercel’in verdiği adresi buraya ekleyeceksin (örn. `https://nad-pwa.vercel.app`).

7. **Create Web Service** de.  
   İlk deploy birkaç dakika sürer. Bittiğinde **Dashboard’da servise tıkla** → üstte **URL** görünür (örn. `https://nad-api.onrender.com`). Bu adresi kopyala; frontend’te kullanacaksın.

**Not:** Render free tier’da uyuyan servis 15 dakika istek almazsa uyur; ilk istekte 30–60 sn uyanma süresi olabilir. Bu bitirme projesi için normaldir.

---

## Bölüm 2: Frontend’i Vercel’de canlıya al

1. **Vercel’e gir**  
   [vercel.com](https://vercel.com) → Sign up / Login (GitHub ile).

2. **Yeni proje**  
   **Add New** → **Project** → Aynı GitHub repoyu seç.

3. **Ayarlar**  
   - **Framework Preset:** Vite.  
   - **Root Directory:** `frontend` seç (önemli).  
   - **Build Command:** `npm run build`.  
   - **Output Directory:** `dist`.  
   - **Install Command:** `npm install`.

4. **Environment variables**  
   **Add** ile ekle:  
   - **Name:** `VITE_API_URL`  
   - **Value:** Render’dan kopyaladığın backend adresi (örn. `https://nad-api.onrender.com`).  
   Sonunda `/` olmasın; sadece kök adres: `https://nad-api.onrender.com`.

5. **Deploy** tıkla.  
   Build biterken Vercel bir URL verir (örn. `https://b2-xxx.vercel.app`).

6. **CORS’u güncelle**  
   Vercel’in verdiği frontend adresi (örn. `https://b2-xxx.vercel.app`) belli olduktan sonra Render’a dön → Web Service → **Environment** → **CORS_ORIGINS** değişkenine bu adresi ekle (virgülle ayırarak birden fazla ekleyebilirsin). Servisi **Manual Deploy** ile yeniden başlat.

---

## Bölüm 3: Kontrol

- Tarayıcıda **Vercel URL’ini** aç (PWA).  
- “Health kontrol” ve “Skor test” butonlarına tıkla.  
- Health’te `{"status":"ok", ...}` ve skor testinde bir skor listesi görmelisin.  
- İstersen Chrome’da **Uygulamayı yükle** ile PWA’yı masaüstüne kurabilirsin; canlı URL’den de kurulur.

---

## Özet

| Ne | Nerede | Sonuç |
|----|--------|--------|
| Backend | Render → Web Service | `https://nad-api.onrender.com` gibi bir URL |
| Frontend (PWA) | Vercel → Project, root: `frontend` | `https://xxx.vercel.app` gibi bir URL |
| Bağlantı | Vercel’de `VITE_API_URL` = Render URL | PWA canlıda API’ye bağlanır |

Takıldığın adımı (Render mı, Vercel mi, CORS mu) yazarsan o adımı birlikte netleştiririz.
