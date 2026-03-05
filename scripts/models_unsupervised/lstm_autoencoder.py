"""
🧠 LSTM AUTOENCODER - TEMPORAL ANOMALY DETECTION
================================================

Bu modül, zaman serisi bazlı anomali tespiti için LSTM Autoencoder içerir.

AVANTAJLARI:
- IP bazlı davranış analizi
- Slow scan detection
- Session hijacking tespiti
- Low-and-slow attack detection

KULLANIM:
from lstm_autoencoder import LSTManomalyDetector
detector = LSTMAnomalyDetector(sequence_length=10, n_features=12)
detector.train(normal_traffic)
anomalies = detector.predict(test_traffic)
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, RepeatVector, TimeDistributed, Dense, 
    Dropout, Input, Bidirectional
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class LSTMAnomalyDetector:
    """
    LSTM Autoencoder ile zaman serisi anomali tespiti
    """
    
    def __init__(self, sequence_length=10, n_features=12, 
                 latent_dim=16, dropout_rate=0.2):
        """
        Args:
            sequence_length: Kaç zaman adımı kullanılacak (sliding window)
            n_features: Feature sayısı
            latent_dim: Bottleneck boyutu
            dropout_rate: Dropout oranı
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.history = None
        
    def build_model(self, lstm_units=[64, 32]):
        """
        LSTM Autoencoder modelini oluştur
        """
        # Input
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # ENCODER
        encoded = inputs
        for i, units in enumerate(lstm_units):
            return_sequences = (i < len(lstm_units) - 1)
            encoded = LSTM(
                units, 
                activation='relu',
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                name=f'encoder_lstm_{i+1}'
            )(encoded)
        
        # BOTTLENECK
        bottleneck = Dense(self.latent_dim, activation='relu', 
                          name='bottleneck')(encoded)
        
        # REPEAT VECTOR
        repeated = RepeatVector(self.sequence_length)(bottleneck)
        
        # DECODER
        decoded = repeated
        for i, units in enumerate(reversed(lstm_units)):
            decoded = LSTM(
                units,
                activation='relu',
                return_sequences=True,
                dropout=self.dropout_rate,
                name=f'decoder_lstm_{i+1}'
            )(decoded)
        
        # OUTPUT
        outputs = TimeDistributed(
            Dense(self.n_features),
            name='output'
        )(decoded)
        
        # Model
        self.model = Model(inputs, outputs, name='lstm_autoencoder')
        self.model.compile(optimizer='adam', loss='mse')
        
        print("✅ LSTM Autoencoder Model Built")
        print(f"   Sequence Length: {self.sequence_length}")
        print(f"   Features: {self.n_features}")
        print(f"   Latent Dim: {self.latent_dim}")
        print(f"   Total Parameters: {self.model.count_params():,}")
        
        return self.model
    
    def build_bidirectional_model(self, lstm_units=[64, 32]):
        """
        Bidirectional LSTM Autoencoder (daha güçlü)
        """
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # ENCODER (Bidirectional)
        encoded = inputs
        for i, units in enumerate(lstm_units):
            return_sequences = (i < len(lstm_units) - 1)
            encoded = Bidirectional(
                LSTM(
                    units,
                    activation='relu',
                    return_sequences=return_sequences,
                    dropout=self.dropout_rate
                ),
                name=f'bi_encoder_{i+1}'
            )(encoded)
        
        # BOTTLENECK
        bottleneck = Dense(self.latent_dim, activation='relu')(encoded)
        repeated = RepeatVector(self.sequence_length)(bottleneck)
        
        # DECODER (Bidirectional)
        decoded = repeated
        for i, units in enumerate(reversed(lstm_units)):
            decoded = Bidirectional(
                LSTM(
                    units,
                    activation='relu',
                    return_sequences=True,
                    dropout=self.dropout_rate
                ),
                name=f'bi_decoder_{i+1}'
            )(decoded)
        
        # OUTPUT
        outputs = TimeDistributed(Dense(self.n_features))(decoded)
        
        self.model = Model(inputs, outputs, name='bilstm_autoencoder')
        self.model.compile(optimizer='adam', loss='mse')
        
        print("✅ Bidirectional LSTM Autoencoder Built")
        return self.model
    
    def create_sequences(self, data, group_by_ip=None):
        """
        Sliding window ile sequence oluştur
        
        Args:
            data: numpy array or pandas dataframe
            group_by_ip: DataFrame ise, IP bazlı grouping için column name
        
        Returns:
            sequences: (n_samples, sequence_length, n_features)
        """
        if isinstance(data, pd.DataFrame):
            if group_by_ip:
                # IP bazlı sequence oluştur
                return self._create_ip_based_sequences(data, group_by_ip)
            else:
                data = data.values
        
        # Basit sliding window
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i + self.sequence_length])
        
        return np.array(sequences)
    
    def _create_ip_based_sequences(self, df, ip_column):
        """
        Her IP için ayrı sequence oluştur (daha iyi temporal learning)
        """
        sequences = []
        
        for ip, group in df.groupby(ip_column):
            group_data = group.drop(columns=[ip_column]).values
            
            # Bu IP için yeterli veri varsa
            if len(group_data) >= self.sequence_length:
                for i in range(len(group_data) - self.sequence_length + 1):
                    sequences.append(group_data[i:i + self.sequence_length])
        
        return np.array(sequences) if sequences else np.array([])
    
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
            self.build_model()
        
        # Veriyi scale et
        X_train_flat = X_train.reshape(-1, self.n_features)
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        
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
                'lstm_autoencoder_best.weights.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=0
            )
        ]
        
        # Eğitim
        print(f"\n🚀 Training LSTM Autoencoder...")
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
            axis=(1, 2)
        )
        self.threshold = np.percentile(val_mse, 99)
        
        print(f"   Optimal Threshold: {self.threshold:.6f}")
        
        return self.history
    
    def predict_anomaly_scores(self, X_test):
        """
        Anomaly score hesapla (reconstruction error)
        """
        # Scale
        X_test_flat = X_test.reshape(-1, self.n_features)
        X_test_scaled = self.scaler.transform(X_test_flat)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        # Reconstruction
        X_reconstructed = self.model.predict(X_test_scaled, verbose=0)
        
        # MSE (her sequence için)
        mse_scores = np.mean(
            np.square(X_test_scaled - X_reconstructed),
            axis=(1, 2)
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
        from sklearn.metrics import (
            classification_report, confusion_matrix,
            roc_auc_score, precision_recall_curve
        )
        
        predictions, scores = self.predict(X_test, threshold)
        
        print("\n📊 LSTM Autoencoder Evaluation")
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
            roc_auc = roc_auc_score(y_test, scores)
            print(f"\nROC-AUC Score: {roc_auc:.4f}")
        
        return {
            'confusion_matrix': cm,
            'predictions': predictions,
            'scores': scores
        }
    
    def plot_training_history(self):
        """
        Eğitim history'sini görselleştir
        """
        if self.history is None:
            print("❌ No training history available")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        
        ax.plot(self.history.history['loss'], label='Training Loss')
        ax.plot(self.history.history['val_loss'], label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('LSTM Autoencoder Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('lstm_training_history.png', dpi=300, bbox_inches='tight')
        print("✅ Training history plot saved: lstm_training_history.png")
        
        return fig
    
    def plot_reconstruction_error_distribution(self, X_normal, X_anomaly=None):
        """
        Reconstruction error dağılımını görselleştir
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Normal traffic errors
        normal_scores = self.predict_anomaly_scores(X_normal)
        ax.hist(normal_scores, bins=50, alpha=0.7, label='Normal', 
                color='blue', edgecolor='black')
        
        # Anomaly traffic errors (eğer varsa)
        if X_anomaly is not None:
            anomaly_scores = self.predict_anomaly_scores(X_anomaly)
            ax.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly',
                   color='red', edgecolor='black')
        
        # Threshold line
        if self.threshold:
            ax.axvline(self.threshold, color='green', linestyle='--',
                      linewidth=2, label=f'Threshold ({self.threshold:.4f})')
        
        ax.set_xlabel('Reconstruction Error (MSE)')
        ax.set_ylabel('Frequency')
        ax.set_title('LSTM Autoencoder - Reconstruction Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('lstm_error_distribution.png', dpi=300, bbox_inches='tight')
        print("✅ Error distribution plot saved: lstm_error_distribution.png")
        
        return fig
    
    def save_model(self, model_path='lstm_autoencoder.keras', 
                   scaler_path='lstm_scaler.pkl'):
        """
        Model ve scaler'ı kaydet
        """
        import pickle
        
        self.model.save(model_path)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Threshold'u da kaydet
        config = {
            'threshold': self.threshold,
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'latent_dim': self.latent_dim
        }
        
        with open('lstm_config.pkl', 'wb') as f:
            pickle.dump(config, f)
        
        print(f"✅ Model saved: {model_path}")
        print(f"✅ Scaler saved: {scaler_path}")
        print(f"✅ Config saved: lstm_config.pkl")
    
    @classmethod
    def load_model(cls, model_path='lstm_autoencoder.keras',
                   scaler_path='lstm_scaler.pkl',
                   config_path='lstm_config.pkl'):
        """
        Kaydedilmiş modeli yükle
        """
        import pickle
        from tensorflow.keras.models import load_model
        
        # Config yükle
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        # Instance oluştur
        detector = cls(
            sequence_length=config['sequence_length'],
            n_features=config['n_features'],
            latent_dim=config['latent_dim']
        )
        
        # Model yükle
        detector.model = load_model(model_path)
        
        # Scaler yükle
        with open(scaler_path, 'rb') as f:
            detector.scaler = pickle.load(f)
        
        detector.threshold = config['threshold']
        
        print("✅ LSTM Autoencoder loaded successfully")
        
        return detector


# =============================================================================
# KARŞILAŞTIRMA: STANDARD AE vs LSTM AE
# =============================================================================

class ModelComparator:
    """
    Standard Autoencoder ile LSTM Autoencoder'ı karşılaştır
    """
    
    def __init__(self, standard_ae, lstm_ae):
        self.standard_ae = standard_ae
        self.lstm_ae = lstm_ae
        
    def compare_on_temporal_attacks(self, X_test, y_test, attack_types):
        """
        Temporal saldırılarda performans karşılaştırması
        
        Args:
            X_test: Test verileri
            y_test: Gerçek etiketler
            attack_types: Her sample için saldırı tipi
        """
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        # Her model için tahmin
        _, std_scores = self.standard_ae.predict(X_test)
        _, lstm_scores = self.lstm_ae.predict(X_test)
        
        std_preds = (std_scores > self.standard_ae.threshold).astype(int)
        lstm_preds = (lstm_scores > self.lstm_ae.threshold).astype(int)
        
        # Saldırı tipine göre analiz
        results = []
        
        for attack_type in np.unique(attack_types):
            mask = (attack_types == attack_type)
            
            if mask.sum() == 0:
                continue
            
            y_true_subset = y_test[mask]
            
            std_subset = std_preds[mask]
            lstm_subset = lstm_preds[mask]
            
            results.append({
                'attack_type': attack_type,
                'count': mask.sum(),
                'std_ae_f1': f1_score(y_true_subset, std_subset),
                'lstm_ae_f1': f1_score(y_true_subset, lstm_subset),
                'std_ae_precision': precision_score(y_true_subset, std_subset),
                'lstm_ae_precision': precision_score(y_true_subset, lstm_subset),
                'std_ae_recall': recall_score(y_true_subset, std_subset),
                'lstm_ae_recall': recall_score(y_true_subset, lstm_subset)
            })
        
        df_results = pd.DataFrame(results)
        
        print("\n🔄 MODEL COMPARISON - BY ATTACK TYPE")
        print("=" * 80)
        print(df_results.to_string(index=False))
        
        # Visualization
        self._plot_comparison(df_results)
        
        return df_results
    
    def _plot_comparison(self, df_results):
        """
        Karşılaştırma grafiği
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        metrics = ['f1', 'precision', 'recall']
        titles = ['F1-Score', 'Precision', 'Recall']
        
        for ax, metric, title in zip(axes, metrics, titles):
            x = np.arange(len(df_results))
            width = 0.35
            
            ax.bar(x - width/2, df_results[f'std_ae_{metric}'], 
                   width, label='Standard AE', color='skyblue')
            ax.bar(x + width/2, df_results[f'lstm_ae_{metric}'],
                   width, label='LSTM AE', color='salmon')
            
            ax.set_xlabel('Attack Type')
            ax.set_ylabel(title)
            ax.set_title(f'{title} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(df_results['attack_type'], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✅ Comparison plot saved: model_comparison.png")


# =============================================================================
# KULLANIM ÖRNEĞİ
# =============================================================================

if __name__ == "__main__":
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║         LSTM AUTOENCODER - TEMPORAL ANOMALY DETECTION      ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Örnek veri (gerçek kullanımda kendi verilerinizi yükleyin)
    print("\n1️⃣ Creating example data...")
    n_samples = 1000
    n_features = 12
    sequence_length = 10
    
    # Normal trafik simülasyonu
    X_normal = np.random.randn(n_samples, n_features) * 0.5
    
    # Anomali simülasyonu (temporal pattern ile)
    X_anomaly = np.random.randn(100, n_features) * 2
    
    # LSTM Detector oluştur
    print("\n2️⃣ Building LSTM Autoencoder...")
    detector = LSTMAnomalyDetector(
        sequence_length=sequence_length,
        n_features=n_features,
        latent_dim=16
    )
    
    detector.build_model(lstm_units=[64, 32])
    
    # Sequence oluştur
    print("\n3️⃣ Creating sequences...")
    X_train_seq = detector.create_sequences(X_normal)
    print(f"   Training sequences shape: {X_train_seq.shape}")
    
    # Eğit
    print("\n4️⃣ Training model...")
    detector.train(
        X_train_seq,
        epochs=20,
        batch_size=32,
        patience=5,
        verbose=1
    )
    
    # Test
    print("\n5️⃣ Testing...")
    X_test_seq = detector.create_sequences(X_anomaly)
    predictions, scores = detector.predict(X_test_seq)
    
    print(f"\n   Anomalies detected: {predictions.sum()}/{len(predictions)}")
    
    # Görselleştir
    print("\n6️⃣ Creating visualizations...")
    detector.plot_training_history()
    detector.plot_reconstruction_error_distribution(X_train_seq, X_test_seq)
    
    # Kaydet
    print("\n7️⃣ Saving model...")
    detector.save_model()
    
    print("\n✅ LSTM Autoencoder demo completed!")
    print("\nNext steps:")
    print("  1. Gerçek conn.log verilerinizi yükleyin")
    print("  2. IP bazlı grouping kullanın")
    print("  3. Standard AE ile karşılaştırın")
