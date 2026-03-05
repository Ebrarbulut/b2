"""
⚙️ HYPERPARAMETER TUNING SİSTEMİ
==================================

GridSearch ve RandomSearch ile model hyperparameter optimizasyonu.

KULLANIM:
from hyperparameter_tuning import HyperparameterTuner
tuner = HyperparameterTuner()
best_params = tuner.tune_isolation_forest(X_train, y_train)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import warnings
warnings.filterwarnings('ignore')

class HyperparameterTuner:
    """
    Model hyperparameter tuning sistemi
    """
    
    def __init__(self, cv=5, scoring='f1', n_jobs=-1, random_state=42):
        """
        Args:
            cv: Cross-validation fold sayısı
            scoring: Scoring metric ('f1', 'roc_auc', 'precision', 'recall')
            n_jobs: Paralel işleme sayısı
            random_state: Random seed
        """
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # Scoring function
        if scoring == 'f1':
            self.scoring = make_scorer(f1_score, greater_is_better=True)
        elif scoring == 'roc_auc':
            self.scoring = make_scorer(roc_auc_score, greater_is_better=True, needs_proba=False)
        else:
            self.scoring = scoring
    
    def tune_isolation_forest(self, X_train, y_train, method='grid', n_iter=50):
        """
        Isolation Forest hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training labels
            method: 'grid' veya 'random'
            n_iter: RandomSearch için iteration sayısı
        """
        print("\n🌲 Tuning Isolation Forest...")
        
        # Parameter grid
        param_grid = {
            'contamination': [0.05, 0.1, 0.15, 0.2],
            'n_estimators': [50, 100, 200],
            'max_samples': ['auto', 0.5, 0.8],
            'max_features': [0.5, 0.8, 1.0]
        }
        
        # Model
        model = IsolationForest(random_state=self.random_state)
        
        # Search
        if method == 'grid':
            search = GridSearchCV(
                model, param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=1
            )
        else:
            search = RandomizedSearchCV(
                model, param_grid,
                n_iter=n_iter,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=1
            )
        
        # Fit
        search.fit(X_train)
        
        print(f"\n✅ Best parameters: {search.best_params_}")
        print(f"   Best score: {search.best_score_:.4f}")
        
        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_model': search.best_estimator_,
            'cv_results': pd.DataFrame(search.cv_results_)
        }
    
    def tune_one_class_svm(self, X_train, y_train, method='grid', n_iter=50):
        """
        One-Class SVM hyperparameter tuning
        """
        print("\n🔷 Tuning One-Class SVM...")
        
        # Parameter grid
        param_grid = {
            'nu': [0.05, 0.1, 0.15, 0.2, 0.25],
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }
        
        # Model
        model = OneClassSVM(random_state=self.random_state)
        
        # Search
        if method == 'grid':
            search = GridSearchCV(
                model, param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=1
            )
        else:
            search = RandomizedSearchCV(
                model, param_grid,
                n_iter=n_iter,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=1
            )
        
        # Fit
        search.fit(X_train)
        
        print(f"\n✅ Best parameters: {search.best_params_}")
        print(f"   Best score: {search.best_score_:.4f}")
        
        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_model': search.best_estimator_,
            'cv_results': pd.DataFrame(search.cv_results_)
        }
    
    def tune_lstm_autoencoder(self, X_train, y_train, 
                             sequence_length=10, method='random', n_iter=20):
        """
        LSTM Autoencoder hyperparameter tuning
        
        Not: LSTM için GridSearch çok yavaş olabilir, RandomSearch önerilir
        """
        print("\n🧠 Tuning LSTM Autoencoder...")
        print("⚠️  Bu işlem uzun sürebilir!")
        
        from scripts.lstm_autoencoder import LSTMAnomalyDetector
        
        # Parameter grid
        param_grid = {
            'sequence_length': [5, 10, 15],
            'latent_dim': [8, 16, 32],
            'lstm_units': [
                [32, 16],
                [64, 32],
                [128, 64]
            ],
            'dropout_rate': [0.1, 0.2, 0.3]
        }
        
        results = []
        
        # Manual search (LSTM için GridSearch çok yavaş)
        if method == 'random':
            from random import choice
            
            for i in range(n_iter):
                print(f"\n  Iteration {i+1}/{n_iter}...")
                
                # Random parameter seç
                params = {
                    'sequence_length': choice(param_grid['sequence_length']),
                    'latent_dim': choice(param_grid['latent_dim']),
                    'lstm_units': choice(param_grid['lstm_units']),
                    'dropout_rate': choice(param_grid['dropout_rate'])
                }
                
                try:
                    # Model oluştur
                    detector = LSTMAnomalyDetector(
                        sequence_length=params['sequence_length'],
                        n_features=X_train.shape[1],
                        latent_dim=params['latent_dim'],
                        dropout_rate=params['dropout_rate']
                    )
                    
                    detector.build_model(lstm_units=params['lstm_units'])
                    
                    # Sequence oluştur
                    X_train_seq = detector.create_sequences(X_train)
                    
                    if len(X_train_seq) == 0:
                        continue
                    
                    # Eğit (kısa epoch)
                    detector.train(
                        X_train_seq,
                        epochs=10,  # Hızlı test için
                        batch_size=32,
                        patience=3,
                        verbose=0
                    )
                    
                    # Cross-validation score (basit)
                    from sklearn.model_selection import cross_val_score
                    # Not: LSTM için CV karmaşık, basit train/test split kullan
                    X_train_cv, X_val_cv = train_test_split(
                        X_train_seq, test_size=0.2, random_state=42
                    )
                    
                    val_scores = detector.predict_anomaly_scores(X_val_cv)
                    # Basit scoring (gerçek CV yerine)
                    score = np.mean(val_scores)  # Placeholder
                    
                    results.append({
                        'params': params,
                        'score': score
                    })
                    
                    print(f"    Score: {score:.4f}")
                    
                except Exception as e:
                    print(f"    ❌ Error: {e}")
                    continue
        
        # En iyi parametreleri bul
        if results:
            best_result = max(results, key=lambda x: x['score'])
            print(f"\n✅ Best parameters: {best_result['params']}")
            print(f"   Best score: {best_result['score']:.4f}")
            
            return {
                'best_params': best_result['params'],
                'best_score': best_result['score'],
                'all_results': results
            }
        else:
            print("❌ No successful iterations!")
            return None
    
    def compare_tuned_models(self, X_train, y_train, X_test, y_test):
        """
        Tüm modelleri tune et ve karşılaştır
        """
        print("\n" + "="*70)
        print("⚙️ COMPREHENSIVE HYPERPARAMETER TUNING")
        print("="*70)
        
        results = {}
        
        # 1. Isolation Forest
        try:
            if_result = self.tune_isolation_forest(X_train, y_train, method='random', n_iter=20)
            if if_result:
                # Test et
                from scripts.isolation_forest_detector import IsolationForestDetector
                best_params = if_result['best_params']
                
                detector = IsolationForestDetector(
                    contamination=best_params['contamination'],
                    n_estimators=best_params['n_estimators'],
                    random_state=self.random_state
                )
                detector.train(X_train[y_train == 0])
                predictions, scores = detector.predict(X_test)
                
                from sklearn.metrics import f1_score
                results['Isolation Forest (Tuned)'] = {
                    'f1': f1_score(y_test, predictions),
                    'params': best_params
                }
        except Exception as e:
            print(f"❌ Isolation Forest tuning failed: {e}")
        
        # 2. One-Class SVM
        try:
            ocsvm_result = self.tune_one_class_svm(X_train, y_train, method='random', n_iter=20)
            if ocsvm_result:
                from scripts.one_class_svm_detector import OneClassSVMDetector
                best_params = ocsvm_result['best_params']
                
                detector = OneClassSVMDetector(
                    nu=best_params['nu'],
                    kernel=best_params['kernel'],
                    gamma=best_params['gamma'],
                    random_state=self.random_state
                )
                detector.train(X_train[y_train == 0])
                predictions, scores = detector.predict(X_test)
                
                from sklearn.metrics import f1_score
                results['One-Class SVM (Tuned)'] = {
                    'f1': f1_score(y_test, predictions),
                    'params': best_params
                }
        except Exception as e:
            print(f"❌ One-Class SVM tuning failed: {e}")
        
        # Sonuçları yazdır
        print("\n" + "="*70)
        print("📊 TUNING RESULTS")
        print("="*70)
        
        for model_name, result in results.items():
            print(f"\n{model_name}:")
            print(f"  F1-Score: {result['f1']:.4f}")
            print(f"  Best Params: {result['params']}")
        
        return results


# =============================================================================
# KULLANIM ÖRNEĞİ
# =============================================================================

if __name__ == "__main__":
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║           HYPERPARAMETER TUNING SYSTEM                     ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Örnek veri
    from sklearn.model_selection import train_test_split
    
    n_samples = 1000
    n_features = 12
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Tuner oluştur
    tuner = HyperparameterTuner(cv=3, scoring='f1')
    
    # Tune et
    print("\n1️⃣ Tuning Isolation Forest...")
    if_result = tuner.tune_isolation_forest(X_train, y_train, method='random', n_iter=10)
    
    print("\n2️⃣ Tuning One-Class SVM...")
    ocsvm_result = tuner.tune_one_class_svm(X_train, y_train, method='random', n_iter=10)
    
    print("\n✅ Hyperparameter tuning completed!")
