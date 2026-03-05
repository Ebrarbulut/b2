"""
🎯 ENSEMBLE ANOMALY DETECTION
==============================

Birden fazla modeli birleştirerek daha güçlü anomali tespiti.

KULLANIM:
from ensemble_detector import EnsembleDetector
ensemble = EnsembleDetector()
ensemble.add_model('isolation_forest', if_model)
ensemble.add_model('one_class_svm', ocsvm_model)
predictions, scores = ensemble.predict(X_test, method='voting')
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

class EnsembleDetector:
    """
    Ensemble anomali tespit sistemi
    """
    
    def __init__(self):
        """
        Ensemble detector oluştur
        """
        self.models = {}
        self.model_weights = {}
    
    def add_model(self, name, model, weight=1.0):
        """
        Modele ensemble'a ekle
        
        Args:
            name: Model ismi
            model: Model instance (predict metodu olmalı)
            weight: Model ağırlığı (voting için)
        """
        self.models[name] = model
        self.model_weights[name] = weight
        print(f"✅ Model '{name}' added (weight: {weight})")
    
    def remove_model(self, name):
        """
        Modeli ensemble'dan çıkar
        """
        if name in self.models:
            del self.models[name]
            del self.model_weights[name]
            print(f"✅ Model '{name}' removed")
        else:
            print(f"❌ Model '{name}' not found")
    
    def voting_predict(self, X_test, threshold=0.5):
        """
        Voting ensemble: Modellerin çoğunluğu
        
        Args:
            X_test: Test features
            threshold: Kaç model anomali derse anomali sayılır (0-1 arası)
        
        Returns:
            predictions: 0 (normal) veya 1 (anomaly)
            scores: Voting scores (0-1 arası, kaç model anomali dedi)
        """
        if not self.models:
            raise ValueError("No models in ensemble!")
        
        all_predictions = []
        all_scores = []
        
        # Her model için tahmin
        for name, model in self.models.items():
            try:
                pred, scores = model.predict(X_test)
                all_predictions.append(pred)
                all_scores.append(scores)
            except Exception as e:
                print(f"⚠️ Model '{name}' prediction failed: {e}")
                continue
        
        if not all_predictions:
            raise ValueError("All models failed!")
        
        # Stack predictions
        all_predictions = np.array(all_predictions)
        all_scores = np.array(all_scores)
        
        # Voting: Kaç model anomali dedi?
        vote_scores = np.mean(all_predictions, axis=0)
        
        # Threshold'a göre final prediction
        final_predictions = (vote_scores >= threshold).astype(int)
        
        return final_predictions, vote_scores
    
    def weighted_voting_predict(self, X_test, threshold=0.5):
        """
        Weighted voting: Ağırlıklı oylama
        """
        if not self.models:
            raise ValueError("No models in ensemble!")
        
        weighted_predictions = np.zeros(len(X_test))
        total_weight = 0
        
        # Her model için ağırlıklı tahmin
        for name, model in self.models.items():
            try:
                pred, scores = model.predict(X_test)
                weight = self.model_weights[name]
                weighted_predictions += pred * weight
                total_weight += weight
            except Exception as e:
                print(f"⚠️ Model '{name}' prediction failed: {e}")
                continue
        
        if total_weight == 0:
            raise ValueError("All models failed!")
        
        # Normalize
        weighted_scores = weighted_predictions / total_weight
        
        # Threshold'a göre final prediction
        final_predictions = (weighted_scores >= threshold).astype(int)
        
        return final_predictions, weighted_scores
    
    def stacking_predict(self, X_test, meta_model=None):
        """
        Stacking: Meta-model ile birleştirme
        
        Args:
            X_test: Test features
            meta_model: Meta-learner (opsiyonel, yoksa basit voting)
        """
        if not self.models:
            raise ValueError("No models in ensemble!")
        
        # Base model predictions (meta-features)
        meta_features = []
        
        for name, model in self.models.items():
            try:
                pred, scores = model.predict(X_test)
                meta_features.append(scores)  # Scores kullan
            except Exception as e:
                print(f"⚠️ Model '{name}' prediction failed: {e}")
                continue
        
        if not meta_features:
            raise ValueError("All models failed!")
        
        meta_features = np.array(meta_features).T  # (n_samples, n_models)
        
        # Meta-model varsa kullan
        if meta_model is not None:
            try:
                final_predictions = meta_model.predict(meta_features)
                final_scores = meta_model.predict_proba(meta_features)[:, 1]
                return final_predictions, final_scores
            except:
                print("⚠️ Meta-model failed, using simple voting")
        
        # Yoksa basit voting
        vote_scores = np.mean(meta_features, axis=1)
        final_predictions = (vote_scores > np.median(vote_scores)).astype(int)
        
        return final_predictions, vote_scores
    
    def average_scores_predict(self, X_test, threshold=None):
        """
        Score averaging: Tüm modellerin skorlarını ortala
        """
        if not self.models:
            raise ValueError("No models in ensemble!")
        
        all_scores = []
        
        for name, model in self.models.items():
            try:
                pred, scores = model.predict(X_test)
                all_scores.append(scores)
            except Exception as e:
                print(f"⚠️ Model '{name}' prediction failed: {e}")
                continue
        
        if not all_scores:
            raise ValueError("All models failed!")
        
        # Average scores
        avg_scores = np.mean(all_scores, axis=0)
        
        # Threshold
        if threshold is None:
            threshold = np.percentile(avg_scores, 95)
        
        final_predictions = (avg_scores > threshold).astype(int)
        
        return final_predictions, avg_scores
    
    def predict(self, X_test, method='voting', **kwargs):
        """
        Ensemble prediction
        
        Args:
            X_test: Test features
            method: 'voting', 'weighted_voting', 'stacking', 'average_scores'
            **kwargs: Method-specific parameters
        """
        if method == 'voting':
            return self.voting_predict(X_test, **kwargs)
        elif method == 'weighted_voting':
            return self.weighted_voting_predict(X_test, **kwargs)
        elif method == 'stacking':
            return self.stacking_predict(X_test, **kwargs)
        elif method == 'average_scores':
            return self.average_scores_predict(X_test, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def evaluate(self, X_test, y_test, method='voting', **kwargs):
        """
        Ensemble performansını değerlendir
        """
        predictions, scores = self.predict(X_test, method=method, **kwargs)
        
        results = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, zero_division=0),
            'recall': recall_score(y_test, predictions, zero_division=0),
            'f1': f1_score(y_test, predictions, zero_division=0),
            'predictions': predictions,
            'scores': scores
        }
        
        print("\n📊 Ensemble Evaluation")
        print("=" * 60)
        print(f"Method: {method}")
        print(f"Models: {', '.join(self.models.keys())}")
        print(f"\nAccuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1-Score: {results['f1']:.4f}")
        
        return results
    
    def compare_methods(self, X_test, y_test):
        """
        Tüm ensemble metodlarını karşılaştır
        """
        print("\n" + "="*70)
        print("🔄 ENSEMBLE METHODS COMPARISON")
        print("="*70)
        
        methods = ['voting', 'weighted_voting', 'average_scores']
        results = {}
        
        for method in methods:
            print(f"\n📊 Testing {method}...")
            try:
                result = self.evaluate(X_test, y_test, method=method)
                results[method] = {
                    'f1': result['f1'],
                    'accuracy': result['accuracy'],
                    'precision': result['precision'],
                    'recall': result['recall']
                }
            except Exception as e:
                print(f"❌ {method} failed: {e}")
                results[method] = None
        
        # Sonuçları yazdır
        print("\n" + "="*70)
        print("📊 COMPARISON RESULTS")
        print("="*70)
        
        comparison_df = pd.DataFrame(results).T
        print(comparison_df)
        
        # En iyi method
        if results:
            best_method = max(
                [m for m in results.keys() if results[m] is not None],
                key=lambda m: results[m]['f1']
            )
            print(f"\n🏆 Best Method: {best_method}")
            print(f"   F1-Score: {results[best_method]['f1']:.4f}")
        
        return results


# =============================================================================
# KULLANIM ÖRNEĞİ
# =============================================================================

if __name__ == "__main__":
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║            ENSEMBLE ANOMALY DETECTION                      ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Örnek veri
    from sklearn.model_selection import train_test_split
    from scripts.isolation_forest_detector import IsolationForestDetector
    from scripts.one_class_svm_detector import OneClassSVMDetector
    
    n_samples = 1000
    n_features = 12
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Modelleri eğit
    print("\n1️⃣ Training individual models...")
    
    if_model = IsolationForestDetector(contamination=0.1)
    if_model.train(X_train[y_train == 0])
    
    ocsvm_model = OneClassSVMDetector(nu=0.1)
    ocsvm_model.train(X_train[y_train == 0])
    
    # Ensemble oluştur
    print("\n2️⃣ Creating ensemble...")
    ensemble = EnsembleDetector()
    ensemble.add_model('isolation_forest', if_model, weight=1.0)
    ensemble.add_model('one_class_svm', ocsvm_model, weight=1.0)
    
    # Metodları karşılaştır
    print("\n3️⃣ Comparing ensemble methods...")
    ensemble.compare_methods(X_test, y_test)
    
    print("\n✅ Ensemble demo completed!")
