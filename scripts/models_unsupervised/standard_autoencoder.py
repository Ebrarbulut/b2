"""
🔷 STANDARD AUTOENCODER - ANOMALY DETECTION
===========================================

Dense layer'lı standart autoencoder. LSTM Autoencoder ile karşılaştırma için.

AVANTAJLARI:
- Hızlı eğitim
- Basit mimari
- Non-temporal veriler için uygun

KULLANIM:
from standard_autoencoder import StandardAutoencoder
detector = StandardAutoencoder(encoding_dim=16)
detector.train(X_train)
predictions, scores = detector.predict(X_test)
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score
)
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

class StandardAutoencoder:
    """
    Standard Dense Autoencoder ile anomali tespiti
    """
    
    def __init__(self, encoding_dim=16, activation='relu', 
                 optimizer='adam', loss='mse'):
        """
        Args:
            encoding_dim: Bottleneck boyutu
            activation: Activation function
            optimizer: Optimizer
            loss: Loss function
        """
        self.encoding_dim = encoding_dim
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        
        self.model = None
        self.encoder = None
        self.decoder = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.history = None
        self.input_dim = None
        
    def build_model(self, input_dim):
        """
        Autoencoder modelini oluştur
        
        Args:
            input_dim: Input feature sayısı
        """
        self.input_dim = input_dim
        
        # Input
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoded = Dense(self.encoding_dim * 2, activation=self.activation)(input_layer)
        encoded = Dense(self.encoding_dim, activation=self.activation)(encoded)
        
        # Decoder
        decoded = Dense(self.encoding_dim * 2, activation=self.activation)(encoded)
        decoded = Dense(input_dim, activation='linear')(decoded)
        
        # Autoencoder model
        self.model = Model(input_layer, decoded, name='standard_autoencoder')
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        
        # Encoder model (ayrı)
        self.encoder = Model(input_layer, encoded, name='encoder')
        
        print("✅ Standard Autoencoder Model Built")
        print(f"   Input Dim: {input_dim}")
        print(f"   Encoding Dim: {self.encoding_dim}")
        print(f"   Total Parameters: {self.model.count_params():,}")
        
        return self.model
    
    def train(self, X_train, validation_split=0.2, epochs=50, 
              batch_size=32, patience=10, verbose=1):
        """
        Modeli eğit
        
        Args:
            X_train: Normal trafik verileri
            validation_split: Validation oranı
            epochs: Epoch sayısı
            batch_size: Batch size
            patience: Early stopping patience
        """
        # Model yoksa oluştur
        if self.model is None:
            self.build_model(X_train.shape[1])
        
        # Veriyi scale et
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Callbacks
        # NOTE (Windows/TF+Keras combos): ModelCheckpoint with native .keras format
        # can error depending on installed Keras version. To be robust, we checkpoint
        # weights only.
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=verbose
            ),
            ModelCheckpoint(
                'standard_autoencoder_best.weights.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=0
            )
        ]
        
        # Eğitim
        print(f"\n🚀 Training Standard Autoencoder...")
        print(f"   Training samples: {len(X_train_scaled)}")
        print(f"   Validation split: {validation_split}")
        
        self.history = self.model.fit(
            X_train_scaled, X_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print("\n✅ Training completed!")
        
        # Threshold belirle (validation loss üzerinden)
        val_predictions = self.model.predict(
            X_train_scaled[int(len(X_train_scaled) * (1 - validation_split)):],
            verbose=0
        )
        val_mse = np.mean(
            np.square(X_train_scaled[int(len(X_train_scaled) * (1 - validation_split)):] 
                     - val_predictions),
            axis=1
        )
        self.threshold = np.percentile(val_mse, 99)
        
        print(f"   Optimal Threshold: {self.threshold:.6f}")
        
        return self.history
    
    def predict_anomaly_scores(self, X_test):
        """
        Anomaly score hesapla (reconstruction error)
        """
        # Scale
        X_test_scaled = self.scaler.transform(X_test)
        
        # Reconstruction
        X_reconstructed = self.model.predict(X_test_scaled, verbose=0)
        
        # MSE (her sample için)
        mse_scores = np.mean(
            np.square(X_test_scaled - X_reconstructed),
            axis=1
        )
        
        return mse_scores
    
    def predict(self, X_test, threshold=None):
        """
        Anomali tespiti yap
        
        Returns:
            predictions: 0 (normal) veya 1 (anomaly)
            scores: reconstruction error scores
        """
        scores = self.predict_anomaly_scores(X_test)
        
        if threshold is None:
            threshold = self.threshold
        
        predictions = (scores > threshold).astype(int)
        
        return predictions, scores
    
    def evaluate(self, X_test, y_test, threshold=None):
        """
        Model performansını değerlendir
        """
        predictions, scores = self.predict(X_test, threshold)
        
        print("\n📊 Standard Autoencoder Evaluation")
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
    
    def plot_training_history(self, save_path='standard_ae_training_history.png'):
        """
        Eğitim history'sini görselleştir
        """
        if self.history is None:
            print("❌ No training history available")
            return
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax.plot(self.history.history['val_loss'], label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Standard Autoencoder Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training history plot saved: {save_path}")
        
        return fig
    
    def save_model(self, model_path='standard_autoencoder.keras', 
                   scaler_path='standard_ae_scaler.pkl'):
        """
        Model ve scaler'ı kaydet
        """
        self.model.save(model_path)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Config kaydet
        config = {
            'encoding_dim': self.encoding_dim,
            'activation': self.activation,
            'optimizer': self.optimizer,
            'loss': self.loss,
            'threshold': self.threshold,
            'input_dim': self.input_dim
        }
        
        with open('standard_ae_config.pkl', 'wb') as f:
            pickle.dump(config, f)
        
        print(f"✅ Model saved: {model_path}")
        print(f"✅ Scaler saved: {scaler_path}")
        print(f"✅ Config saved: standard_ae_config.pkl")
    
    @classmethod
    def load_model(cls, model_path='standard_autoencoder.keras',
                   scaler_path='standard_ae_scaler.pkl',
                   config_path='standard_ae_config.pkl'):
        """
        Kaydedilmiş modeli yükle
        """
        from tensorflow.keras.models import load_model
        
        # Config yükle
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        # Instance oluştur
        detector = cls(
            encoding_dim=config['encoding_dim'],
            activation=config['activation'],
            optimizer=config['optimizer'],
            loss=config['loss']
        )
        
        # Model yükle
        detector.model = load_model(model_path)
        detector.input_dim = config['input_dim']
        
        # Scaler yükle
        with open(scaler_path, 'rb') as f:
            detector.scaler = pickle.load(f)
        
        detector.threshold = config['threshold']
        
        print("✅ Standard Autoencoder loaded successfully")
        
        return detector


# =============================================================================
# KULLANIM ÖRNEĞİ
# =============================================================================

if __name__ == "__main__":
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║         STANDARD AUTOENCODER - ANOMALY DETECTION          ║
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
    print("\n2️⃣ Building Standard Autoencoder...")
    detector = StandardAutoencoder(encoding_dim=16)
    
    # Eğit
    print("\n3️⃣ Training...")
    detector.train(
        X_train_normal,
        epochs=50,
        batch_size=32,
        patience=10,
        verbose=1
    )
    
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
    detector.plot_training_history()
    
    # Kaydet
    print("\n7️⃣ Saving model...")
    detector.save_model()
    
    print("\n✅ Standard Autoencoder demo completed!")
