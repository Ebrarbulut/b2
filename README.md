# Network Anomaly Detection System

A machine learning-based network intrusion detection system using autoencoders (LSTM and Standard) for identifying anomalous network traffic patterns.

## ⚠️ Security & Ethics Disclaimer

> [!CAUTION]
> **This tool is designed EXCLUSIVELY for educational purposes and legitimate cybersecurity defense research.**
>
> **PROHIBITED USES:**
> - Unauthorized network scanning or penetration testing
> - Attacking systems you do not own or have explicit permission to test
> - Any malicious activities or illegal network intrusions
> - Circumventing security measures without authorization
>
> **LEGAL NOTICE:**
> - Users are solely responsible for compliance with all applicable laws and regulations
> - Unauthorized access to computer systems is illegal in most jurisdictions
> - Always obtain proper authorization before testing any network or system
> - The author assumes NO liability for misuse of this software
>
> **USE RESPONSIBLY:** This tool should only be used in controlled environments, for authorized security assessments, or for educational purposes with proper permissions.

## 🎯 Features

- **Dual Autoencoder Models**: LSTM-based and Standard autoencoders for anomaly detection
- **Real-time Analysis**: Streamlit-based web interface for interactive analysis
- **Ensemble Voting**: Combines multiple models for improved detection accuracy
- **PCAP Processing**: Converts network packet captures to Zeek connection logs
- **Feature Engineering**: Automated feature extraction from network traffic
- **Model Optimization**: Hyperparameter tuning and threshold optimization
- **Comprehensive Metrics**: Detailed performance evaluation and visualization

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.12+
- Zeek (for PCAP processing)
- See `requirements.txt` for full dependencies

## 📥 Nasıl indirilir (How to download)

- **Git ile (önerilen):** Bilgisayarında Git kuruluysa:
  ```bash
  git clone https://github.com/Ebrarbulut/b2.git
  cd b2
  ```
- **ZIP ile:** GitHub sayfasında yeşil **Code** → **Download ZIP** ile indir, aç, klasör adı genelde `b2-main` olur; terminalde bu klasöre gir (`cd b2-main`).

İndirdikten sonra kurulum adımlarına (sanal ortam, bağımlılıklar, isteğe bağlı veri setleri) geçebilirsin.

---

## 🚀 Kurulum (Installation)

1. Clone the repository:
```bash
git clone https://github.com/Ebrarbulut/network_anomaly_detection_system.git
cd network_anomaly_detection_system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Zeek (required for PCAP processing):
   - **Windows**: Download from [zeek.org](https://zeek.org/)
   - **Linux**: `sudo apt-get install zeek`
   - **macOS**: `brew install zeek`

## 📊 Kullanım (Usage)

### Web Arayüzü (Streamlit)

Launch the main application:
```bash
streamlit run streamlit_app.py
```

Additional interfaces:
- **Custom Analysis**: `streamlit run streamlit_analyze_custom.py`
- **Ensemble Selector**: `streamlit run streamlit_ensemble_selector.py`
- **Custom Results**: `streamlit run streamlit_custom_results.py`

### Veri İşleme (Data Processing)

1. **Convert PCAP to Connection Logs**:
```bash
python scripts/pcap_to_connlog_simple.py
```

2. **Merge and Label Logs**:
```bash
python scripts/merge_and_label_logs.py
```

### Model Eğitimi (Model Training)

Train the autoencoder models:
```bash
# Standard Autoencoder
python scripts/optimize_autoencoder.py

# LSTM Autoencoder (if available)
python scripts/optimize_lstm_autoencoder.py
```

### Model Değerlendirme (Model Evaluation)

## 📦 Veri Setleri ve Projeyi Çalıştırma (Datasets & Running the Project)

> **Not:** Büyük veri dosyaları (PCAP ve büyük CSV dosyaları) GitHub dosya boyutu limitleri nedeniyle repoda tutulmamaktadır.
> Projeyi çalıştırmak için bu dosyaları harici bir kaynaktan indirip doğru klasörlere yerleştirmeniz gerekir.

### 1. Repoyu klonla

```bash
git clone https://github.com/Ebrarbulut/b2.git
cd b2
```

### 2. Sanal ortam ve bağımlılıklar

```bash
python -m venv venv
# Windows
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Veri setlerini indir ve yerleştir

Büyük veri dosyaları aşağıdaki dosyaları içerir:

- `normal_traffic.pcap`
- `attacks/attack_udpflood.pcap`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-WorkingHours.pcap_ISCX.csv`

Bu dosyaları aşağıdaki Google Drive klasöründen indirebilirsiniz:

[Veri setlerini buradan indirin](https://drive.google.com/drive/folders/1KOVPICYw1dJWRK7Rk8nXbrCkGtX3lNbM?usp=sharing)

İndirdikten sonra proje kök dizininde aşağıdaki konumlara kopyalayın:

```text
data/pcap/normal_traffic.pcap
data/pcap/attacks/attack_udpflood.pcap
data/raw/cicids2017/Wednesday-WorkingHours.pcap_ISCX.csv
data/raw/cicids2017/Tuesday-WorkingHours.pcap_ISCX.csv
```

`data/raw/` ve `data/pcap/` klasörleri `.gitignore` dosyasında tanımlıdır; bu sayede büyük dosyalar GitHub deposuna push edilmez ancak yerel ortamda proje sorunsuz çalışır.

### 4. Streamlit arayüzünü çalıştır

```bash
streamlit run streamlit_app.py
```

Bu komut çalıştıktan sonra tarayıcıda genellikle `http://localhost:8501` adresinden arayüze erişebilirsiniz.

### 5. PWA + API ile çalıştırma (önerilen)

**Backend (API):**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend (React PWA):** Başka bir terminalde:
```bash
cd frontend
npm install
echo "VITE_API_URL=http://localhost:8000" > .env
npm run dev
```
Tarayıcıda `http://localhost:5173` adresini açın. Bağlantıyı kontrol et, örnek analiz, CSV/PCAP yükleme ve model karşılaştırma bu arayüzden yapılır.

**Canlı NIDS (isteğe bağlı):** PowerShell’i **Yönetici olarak** açıp:
```bash
cd B2
.\venv\Scripts\activate
python realtime_nids_scapy.py
```

### 6. Canlı NIDS’i .exe olarak derleme (PowerShell açık kalmadan çalışır)

- **Derleme:** Proje klasöründe:
  ```bash
  scripts\build_nad_sensor_exe.bat
  ```
  Çıktı: `dist\nad_sensor.exe`.

- **Çalıştırma:** `dist\nad_sensor.exe` dosyasına çift tıklayabilir veya kısayol oluşturabilirsin. **PowerShell veya terminal penceresi açılmaz;** uygulama arka planda çalışır. İlk çalıştırmada Windows “Yönetici olarak çalıştır” isteyebilir (paket dinleme yetkisi için).

- **Log:** Durum ve uyarılar `nad_sensor.log` dosyasına yazılır (.exe’nin bulunduğu klasörde veya çalışma dizininde). Takibi oradan yapabilirsin; PowerShell’i sürekli açık tutmana gerek yok.

Compare model performance:
```bash
python scripts/experiments/compare_all_models.py
```

### 7. Testleri çalıştırma

Proje kökünden (backend’e gerek yok, TestClient kullanılır). Sanal ortam kullanıyorsan önce `venv\Scripts\activate` ile aktive edip bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```
Veya script ile: **Windows** `scripts\run_tests.bat`, **Linux/macOS** `bash scripts/run_tests.sh`.  
`tests/test_api_health_score.py` ve `tests/test_analyze_csv_pcap.py` API sağlık, skor ve CSV/PCAP analiz uç noktalarını test eder.

### 8. Piyasaya sürüm / yayına hazırlık

- **API:** Rate limiting açık (IP başına dakikada 60 istek `/api/*`, 120 istek `/health`); 429 yanıtında “Çok fazla istek” mesajı.
- **Frontend:** Açık/koyu tema, okunaklı yazı boyutu ve kalınlık, CSV/PCAP 50 MB limiti, gizlilik notu.
- **Canlı NIDS:** `.exe` ile terminal açılmadan çalışma, log dosyası.
- **Dokümantasyon:** Kurulum, indirme, kullanım ve gizlilik README’de; detaylı analiz `PROJE_DURUM_VE_ANALIZ.md` içinde.

**Bitirme raporu:** Proje kökünde `BITIRME_RAPORU_TASLAK.md` dosyası rapor taslağı içerir; bölümleri doldurup 20–30 sayfaya çıkarabilirsiniz.

## 📁 Proje Yapısı (Project Structure)

```
network_anomaly_detection_system/
├── data/                          # Data directory (gitignored)
│   ├── raw/                       # Raw PCAP and log files
│   ├── labeled/                   # Labeled datasets
│   └── features/                  # Extracted features
├── models/                        # Trained model files (gitignored)
├── outputs/                       # Analysis outputs (gitignored)
├── scripts/                       # Processing and training scripts
│   ├── experiments/               # Model comparison experiments
│   ├── pcap_to_connlog_simple.py # PCAP conversion
│   ├── merge_and_label_logs.py   # Data labeling
│   ├── optimize_autoencoder.py   # Model training
│   └── ensemble_voting.py        # Ensemble methods
├── streamlit_app.py              # Main web interface
├── streamlit_analyze_custom.py   # Custom analysis UI
├── streamlit_ensemble_selector.py # Model selection UI
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🧱 Mimar i Özeti (Architecture Overview)

- **Veri Pipeline'ı**:  
  PCAP → (Zeek veya `pcap_to_connlog_simple.py`) → `conn.log` → `merge_and_label_logs.py` ile etiketleme →
  `data_pipeline/advanced_features.py` ile 12+ özellik çıkarımı → `data/features/advanced_features.csv`.

- **Model Katmanı**:  
  `scripts/models_unsupervised/` altında Standard Autoencoder, LSTM Autoencoder, Isolation Forest, One-Class SVM;  
  `scripts/ensemble_voting.py` ve `monitoring/ensemble_detector.py` ile ensemble ve voting stratejileri.

- **Monitoring & Analiz**:  
  `monitoring/threshold_optimizer.py` ile threshold optimizasyonu,  
  `monitoring/drift_detection.py` ile drift tespiti ve raporlama,  
  `outputs/*.json` raporları üzerinden Streamlit arayüzünde görselleştirme.

- **Arayüz**:  
  `streamlit_app.py` ve diğer `streamlit_*` dosyaları, eğitim/test çıktılarından beslenen
  interaktif güvenlik analizi arayüzü sunar.

## 🔬 Modeller (Models)

### Standard Autoencoder
- Dense neural network architecture
- Efficient for general anomaly detection
- Lower computational requirements

### LSTM Autoencoder
- Recurrent architecture for temporal patterns
- Better for sequential network traffic analysis
- Higher accuracy on time-series data

### Ensemble Voting
- Combines predictions from multiple models
- Reduces false positives
- Configurable voting thresholds

## 📈 Performance Metrics

The system evaluates models using:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC curves
- Confusion matrices
- Reconstruction error distributions

## 🛡️ Security Considerations

- **Veri ve gizlilik:** Web arayüzünde yüklediğiniz CSV/PCAP dosyaları yalnızca analiz sırasında işlenir; sunucuda kalıcı olarak saklanmaz. İstek sonrası yanıt üretilir ve dosya bellekten atılır.
- **Data Privacy**: Ensure all network captures comply with privacy regulations
- **Authorized Testing**: Only analyze traffic from networks you own or have permission to monitor
- **Model Security**: Trained models may contain sensitive information about your network topology
- **Responsible Disclosure**: Report discovered vulnerabilities through proper channels

## 🤝 Contributing

Contributions are welcome for:
- Bug fixes and improvements
- New model architectures
- Enhanced visualization features
- Documentation improvements

Please ensure all contributions align with ethical security research practices.

## 📄 License

This project is provided for educational and research purposes. Users must comply with all applicable laws and regulations.

## 👤 Author

**Ebrar Bulut**
- GitHub: [@Ebrarbulut](https://github.com/Ebrarbulut)

## 🙏 Acknowledgments

- Zeek Network Security Monitor
- TensorFlow and Keras teams
- Streamlit framework
- Open-source cybersecurity community

---

**Remember**: With great power comes great responsibility. Use this tool ethically and legally.
