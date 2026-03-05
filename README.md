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

## 🚀 Installation

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

## 📊 Usage

### Web Interface (Streamlit)

Launch the main application:
```bash
streamlit run streamlit_app.py
```

Additional interfaces:
- **Custom Analysis**: `streamlit run streamlit_analyze_custom.py`
- **Ensemble Selector**: `streamlit run streamlit_ensemble_selector.py`
- **Custom Results**: `streamlit run streamlit_custom_results.py`

### Data Processing

1. **Convert PCAP to Connection Logs**:
```bash
python scripts/pcap_to_connlog_simple.py
```

2. **Merge and Label Logs**:
```bash
python scripts/merge_and_label_logs.py
```

### Model Training

Train the autoencoder models:
```bash
# Standard Autoencoder
python scripts/optimize_autoencoder.py

# LSTM Autoencoder (if available)
python scripts/optimize_lstm_autoencoder.py
```

### Model Evaluation

Compare model performance:
```bash
python scripts/experiments/compare_all_models.py
```

## 📁 Project Structure

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

## 🔬 Models

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
