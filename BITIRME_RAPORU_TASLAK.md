# Ağ Anomali Tespit Sistemi – Bitirme Projesi Rapor Taslağı

**Proje adı:** Network Anomaly Detection System (NADS)  
**Hazırlayan:** [Adınız Soyadınız]  
**Tarih:** [Tarih]  
**Bölüm / Program:** [Bilgisayar Mühendisliği / Siber Güvenlik vb.]

---

## Özet

Bu projede, ağ trafiğindeki anormal davranışları tespit etmek için makine öğrenmesi tabanlı bir sistem geliştirilmiştir. Sistem, sadece normal trafik ile eğitilen denetimsiz modeller (Standard Autoencoder, LSTM Autoencoder, One-Class SVM, Isolation Forest) ve bunların ensemble’ı ile çalışır; hem toplu (CSV/PCAP) hem de gerçek zamanlı analiz sunar. Bir web arayüzü (PWA) ve REST API ile kullanıcıya sunulmuştur.

**Anahtar kelimeler:** Ağ saldırı tespiti, NIDS, anomali tespiti, autoencoder, makine öğrenmesi, denetimsiz öğrenme.

---

## 1. Giriş

### 1.1 Problem ve motivasyon

Ağ tabanlı saldırılar ve anormal trafik, kurumlar için ciddi güvenlik ve gizlilik riski oluşturur. Geleneksel imza tabanlı sistemler bilinen tehditleri yakalayabilir; ancak sıfır gün veya bilinmeyen davranışlar için yetersiz kalabilir. Bu projede, **normal trafik örüntülerini öğrenip** bu örüntülerden sapmaları anomali olarak işaretleyen bir sistem hedeflenmiştir.

### 1.2 Amaç ve kapsam

- Ağ trafiğini akış (flow) bazında özellikle temsil etmek (süre, bayt/paket sayıları, oranlar vb.).
- Birden fazla denetimsiz model (Autoencoder, LSTM AE, One-Class SVM, Isolation Forest) eğitmek ve karşılaştırmak.
- Ensemble yöntemi ile daha kararlı tespit sağlamak.
- Hem toplu (CSV/PCAP) hem de gerçek zamanlı (canlı paket dinleme) analiz sunmak.
- Kullanıcıya web arayüzü (PWA) ve API ile erişilebilir bir ürün sunmak.

### 1.3 Raporun yapısı

Rapor sırasıyla literatür, yöntem, veri ve pipeline, modeller, deneyler, sonuçlar ve tartışma bölümlerinden oluşmaktadır.

---

## 2. Literatür ve ilgili çalışmalar

- **NIDS:** Ağ tabanlı saldırı tespit sistemleri; imza tabanlı ve davranış tabanlı yaklaşımlar.
- **Anomali tespiti:** Denetimsiz öğrenme; normal veri ile eğitim, yeniden yapılandırma hatası veya sınır öğrenme.
- **Autoencoder:** Sıkıştırma ve yeniden yapılandırma; yüksek hata = anomali (Chong ve Tay, 2017; vb.).
- **LSTM Autoencoder:** Zaman serisi ve ardışık davranış; ağ trafiğinde temporal örüntüler (Malhotra vd.).
- **One-Class SVM ve Isolation Forest:** Sadece normal sınıf ile sınır öğrenme; outlier tespiti.

*(Bu bölümü, okuduğunuz makaleler ve kitaplarla genişletebilirsiniz; referans listesi ekleyin.)*

---

## 3. Yöntem

### 3.1 Veri pipeline’ı

1. **Ham veri:** PCAP dosyaları veya Zeek `conn.log` çıktıları.
2. **Akış (flow) tanımı:** Kaynak/hedef IP, port ve protokol (TCP/UDP/ICMP) ile gruplanan bağlantılar.
3. **Etiketleme:** Mevcut veri setlerinde (örn. CICIDS2017) etiket kullanılarak normal/saldırı ayrımı.
4. **Özellik çıkarımı:** Süre, bayt/paket sayıları, oranlar, entropy vb. ile `advanced_features` matrisi (ör. 12+ kolon).
5. **Standardizasyon:** StandardScaler ile ölçekleme (eğitim verisiyle fit, test/canlıda transform).

### 3.2 Modeller

- **Standard Autoencoder:** Dense katmanlı encode–decode; yeniden yapılandırma hatası = anomali skoru.
- **LSTM Autoencoder:** Zaman serisi penceresi (sequence_length); ardışık akışlar üzerinde temporal anomali.
- **One-Class SVM:** Sadece normal veri ile RBF kernel; sınır dışı = anomali.
- **Isolation Forest:** Ağaç tabanlı izolasyon; kısa path = anomali.
- **Ensemble:** Birden fazla modelin oylaması; en az 2 model “anomali” derse uyarı (veya ağırlıklı ortalama).

### 3.3 Gerçek zamanlı NIDS

- Scapy ile paket dinleme; akışlar timeout ve batch boyutuna göre kapatılır.
- Her akış için aynı özellik vektörü hesaplanır; AE (zorunlu) + OCSVM + IF (+ isteğe bağlı LSTM) ile skorlanır.
- Oylama kuralı: en az 2 model anomali derse uyarı; log dosyasına yazılır.

### 3.4 API ve arayüz

- **Backend:** FastAPI; `/health`, `/api/models`, `/api/score`, `/api/analyze-csv`, `/api/analyze-pcap`, `/api/comparison`; rate limiting ve 50 MB dosya limiti.
- **Frontend:** React PWA; tema (açık/koyu), CSV/PCAP yükleme, model listesi ve karşılaştırma tablosu.

---

## 4. Veri seti ve deneysel düzen

- **Veri kaynağı:** CICIDS2017 veya projede kullanılan PCAP/CSV’ler; train/test ayrımı.
- **Özellikler:** `advanced_features` kolonları; eksik değerler 0 veya uygun doldurma.
- **Metrikler:** Accuracy, F1, Precision, Recall, ROC-AUC, FPR; karşılaştırma tablosu `outputs/core_model_comparison.csv` ile üretilir.

*(Bu bölüme kullandığınız veri setinin boyutu, sınıf dağılımı ve deney protokolünü ekleyin.)*

---

## 5. Deneyler ve sonuçlar

### 5.1 Model karşılaştırması

Lokalde `python scripts/experiments/compare_core_models.py` çalıştırıldığında üretilen metrikler (Accuracy, F1, Recall, Precision, ROC-AUC, FPR) tablo ve grafiklerle sunulabilir.

*(Buraya çıktı tablosunu veya ekran görüntüsünü ekleyin.)*

### 5.2 Toplu analiz (CSV/PCAP)

- Örnek CSV veya PCAP ile analiz sonuçları: anomali oranı, skor dağılımı (min/ort/max).
- 50 MB dosya limiti ve rate limiting ile güvenli kullanım.

### 5.3 Canlı NIDS

- .exe veya `python realtime_nids_scapy.py` ile gerçek zamanlı dinleme; uyarılar `nad_sensor.log` içinde.
- Hassasiyet ayarı (yüksek/orta/düşük) ile yanlış pozitif dengesi.

---

## 6. Tartışma ve sınırlamalar

- **Avantajlar:** Çoklu model ve ensemble ile daha kararlı tespit; PWA ve API ile erişilebilirlik; gerçek zamanlı ve toplu analiz.
- **Sınırlamalar:** LSTM canlıda tam entegre değil (opsiyonel eklendi); canlı deploy’da model dosyaları ve `data/features` yönetimi gerekebilir; rate limiting in-memory (dağıtık ortam için Redis vb. düşünülebilir).
- **Etik:** Sadece yetkili ağlarda ve eğitim/araştırma amaçlı kullanım önerilir.

---

## 7. Sonuç ve gelecek çalışmalar

Bu projede, ağ anomali tespiti için uçtan uca bir sistem (veri pipeline’ı, çoklu model, ensemble, API, PWA, canlı NIDS) geliştirilmiş ve dokümante edilmiştir. Sistem bitirme projesi ve demo kullanımı için hazırdır.

**Gelecek çalışmalar:** LSTM’in canlı NIDS’te tam sequence entegrasyonu, kullanıcı kimlik doğrulama, dağıtık rate limiting, daha zengin PWA görselleştirmeleri.

---

## Kaynaklar

*(Okuduğunuz makaleleri ve kitapları buraya ekleyin. Örnek format:)*  
- [1] Y. Chong, S. Tay, “Abnormal event detection in videos using spatiotemporal autoencoder”, LNCS, 2017.  
- [2] CICIDS2017 dataset, Canadian Institute for Cybersecurity.  
- [3] Zeek Network Security Monitor, https://zeek.org  

---

**Not:** Bu dosya bitirme raporu için taslaktır. Bölümleri kendi veri seti sonuçlarınız, ekran görüntüleri ve literatür referanslarınızla genişletip 20–30 sayfa civarına getirebilirsiniz.
