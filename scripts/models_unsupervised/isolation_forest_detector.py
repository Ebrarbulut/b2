"""
🌲 ISOLATION FOREST - ANOMALY DETECTION
========================================

Isolation Forest, anomali tespiti için etkili bir ensemble yöntemidir.
Random forest benzeri ama anomali tespiti için optimize edilmiş.

AVANTAJLARI:
- Hızlı eğitim
- Non-parametric (dağılım varsayımı yok)
- Çok boyutlu verilerde iyi çalışır
- Outlier detection için optimize

KULLANIM:
from isolation_forest_detector import IsolationForestDetector
detector = IsolationForestDetector(contamination=0.1)
detector.train(X_train)
predictions, scores = detector.predict(X_test)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path

class IsolationForestDetector:
    """
    Isolation Forest ile anomali tespiti
    """
    
    def __init__(self, contamination=0.1, n_estimators=100, 
                 max_samples='auto', random_state=42):
        """
        Args:
            contamination: Anomali oranı (0.0-0.5 arası)
            n_estimators: Tree sayısı
            max_samples: Her tree için sample sayısı
            random_state: Random seed
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        
    def train(self, X_train, y_train=None):
        """
        Modeli eğit
        
        Args:
            X_train: Training features (normal trafik)
            y_train: Labels (opsiyonel, unsupervised olduğu için kullanılmaz)
        """
        print(f"\n🌲 Training Isolation Forest...")
        print(f"   Contamination: {self.contamination}")
        print(f"   N Estimators: {self.n_estimators}")
        print(f"   Training samples: {len(X_train)}")
        
        # Normalizasyon
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Model oluştur ve eğit
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1  # Paralel işleme
        )
        
        self.model.fit(X_train_scaled)
        
        # Threshold belirle (decision_function skorlarından)
        train_scores = self.model.decision_function(X_train_scaled)
        self.threshold = np.percentile(train_scores, (1 - self.contamination) * 100)
        
        print(f"✅ Training completed!")
        print(f"   Threshold: {self.threshold:.4f}")
        
        return self
    
    def predict_anomaly_scores(self, X_test):
        """
        Anomaly score hesapla
        
        Returns:
            scores: Negatif değerler = anomali (daha negatif = daha anormal)
        """
        X_test_scaled = self.scaler.transform(X_test)
        scores = self.model.decision_function(X_test_scaled)
        
        # Negatif skorları pozitif yap (anomali = yüksek skor)
        # Isolation Forest: -1 = anomali, +1 = normal
        # Bizim sistem: yüksek skor = anomali
        anomaly_scores = -scores  # Negatifleri pozitif yap
        
        return anomaly_scores
    
    def predict(self, X_test, threshold=None):
        """
        Anomali tespiti yap
        
        Returns:
            predictions: 0 (normal) veya 1 (anomaly)
            scores: anomaly scores (yüksek = anormal)
        """
        scores = self.predict_anomaly_scores(X_test)
        
        if threshold is None:
            threshold = self.threshold
        
        # Isolation Forest: -1 = anomali, +1 = normal
        # Bizim sistem: 1 = anomali, 0 = normal
        if_predictions = self.model.predict(X_test)
        predictions = (if_predictions == -1).astype(int)
        
        return predictions, scores
    
    def evaluate(self, X_test, y_test, threshold=None):
        """
        Model performansını değerlendir
        """
        predictions, scores = self.predict(X_test, threshold)
        
        print("\n📊 Isolation Forest Evaluation")
        print("=" * 60)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, predictions)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(
            y_test, predictions,
            target_names=['Normal', 'Anomaly']
        ))
        
        # ROC-AUC
        if len(np.unique(y_test)) > 1:
            try:
                roc_auc = roc_auc_score(y_test, scores)
                print(f"\nROC-AUC Score: {roc_auc:.4f}")
            except:
                print("\n⚠️ ROC-AUC hesaplanamadı")
        
        return {
            'confusion_matrix': cm,
            'predictions': predictions,
            'scores': scores
        }
    
    def plot_score_distribution(self, X_normal, X_anomaly=None, save_path='if_score_distribution.png'):
        """
        Score dağılımını görselleştir
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Normal traffic scores
        normal_scores = self.predict_anomaly_scores(X_normal)
        ax.hist(normal_scores, bins=50, alpha=0.7, label='Normal', 
                color='green', edgecolor='black')
        
        # Anomaly traffic scores (eğer varsa)
        if X_anomaly is not None:
            anomaly_scores = self.predict_anomaly_scores(X_anomaly)
            ax.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly',
                   color='red', edgecolor='black')
        
        # Threshold line
        if self.threshold:
            ax.axvline(self.threshold, color='blue', linestyle='--',
                      linewidth=2, label=f'Threshold ({self.threshold:.4f})')
        
        ax.set_xlabel('Anomaly Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Isolation Forest - Score Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Score distribution plot saved: {save_path}")
        
        return fig
    
    def save_model(self, model_path='isolation_forest.pkl', 
                   scaler_path='if_scaler.pkl'):
        """
        Model ve scaler'ı kaydet
        """
        # Model kaydet
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Scaler kaydet
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Config kaydet
        config = {
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'threshold': self.threshold,
            'random_state': self.random_state
        }
        
        with open('if_config.pkl', 'wb') as f:
            pickle.dump(config, f)
        
        print(f"✅ Model saved: {model_path}")
        print(f"✅ Scaler saved: {scaler_path}")
        print(f"✅ Config saved: if_config.pkl")
    
    @classmethod
    def load_model(cls, model_path='isolation_forest.pkl',
                   scaler_path='if_scaler.pkl',
                   config_path='if_config.pkl'):
        """
        Kaydedilmiş modeli yükle
        """
        # Config yükle
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        # Instance oluştur
        detector = cls(
            contamination=config['contamination'],
            n_estimators=config['n_estimators'],
            max_samples=config['max_samples'],
            random_state=config['random_state']
        )
        
        # Model yükle
        with open(model_path, 'rb') as f:
            detector.model = pickle.load(f)
        
        # Scaler yükle
        with open(scaler_path, 'rb') as f:
            detector.scaler = pickle.load(f)
        
        detector.threshold = config['threshold']
        
        print("✅ Isolation Forest loaded successfully")
        
        return detector


# =============================================================================
# KULLANIM ÖRNEĞİ
# =============================================================================

if __name__ == "__main__":
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║         ISOLATION FOREST - ANOMALY DETECTION              ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Örnek veri
    print("\n1️⃣ Creating example data...")
    n_samples = 1000
    n_features = 12
    
    # Normal trafik
    X_normal = np.random.randn(n_samples, n_features) * 0.5
    
    # Anomali
    X_anomaly = np.random.randn(100, n_features) * 3
    
    # Birleştir
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([np.zeros(n_samples), np.ones(100)])
    
    # Train/Test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normal trafik ile eğit
    X_train_normal = X_train[y_train == 0]
    
    # Detector oluştur
    print("\n2️⃣ Building Isolation Forest...")
    detector = IsolationForestDetector(
        contamination=0.1,
        n_estimators=100
    )
    
    # Eğit
    print("\n3️⃣ Training...")
    detector.train(X_train_normal)
    
    # Test
    print("\n4️⃣ Testing...")
    predictions, scores = detector.predict(X_test)
    
    print(f"\n   Anomalies detected: {predictions.sum()}/{len(predictions)}")
    print(f"   True anomalies: {y_test.sum()}")
    
    # Değerlendir
    print("\n5️⃣ Evaluation...")
    detector.evaluate(X_test, y_test)
    
    # Görselleştir
    print("\n6️⃣ Creating visualizations...")
    X_test_normal = X_test[y_test == 0]
    X_test_anomaly = X_test[y_test == 1]
    detector.plot_score_distribution(X_test_normal, X_test_anomaly)
    
    # Kaydet
    print("\n7️⃣ Saving model...")
    detector.save_model()
    
    print("\n✅ Isolation Forest demo completed!")
