"""
Model Drift Detection & Monitoring
===================================

Kolmogorov-Smirnov test, Population Stability Index, ve daha fazlası
"""

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import json

class ModelDriftDetector:
    """
    Comprehensive drift detection system
    """
    
    def __init__(self, baseline_errors, window_size=1000, 
                 ks_threshold=0.05, psi_threshold=0.2):
        """
        Args:
            baseline_errors: İlk eğitim sonrası reconstruction errors
            window_size: Rolling window boyutu
            ks_threshold: KS test p-value threshold
            psi_threshold: PSI threshold (> 0.2 = significant drift)
        """
        self.baseline_errors = baseline_errors
        self.window_size = window_size
        self.ks_threshold = ks_threshold
        self.psi_threshold = psi_threshold
        
        self.drift_history = []
        self.monitoring_data = []
    
    def detect_ks_drift(self, current_errors):
        """
        Kolmogorov-Smirnov test
        """
        # Son window_size kadar error al
        if len(current_errors) > self.window_size:
            current_window = current_errors[-self.window_size:]
        else:
            current_window = current_errors
        
        # KS test
        statistic, p_value = stats.ks_2samp(
            self.baseline_errors[-self.window_size:],
            current_window
        )
        
        drift_detected = p_value < self.ks_threshold
        
        result = {
            'method': 'KS-test',
            'statistic': statistic,
            'p_value': p_value,
            'threshold': self.ks_threshold,
            'drift_detected': drift_detected,
            'timestamp': datetime.now().isoformat()
        }
        
        self.drift_history.append(result)
        
        return result
    
    def calculate_psi(self, current_errors, n_bins=10):
        """
        Population Stability Index
        
        PSI < 0.1: No significant drift
        0.1 < PSI < 0.2: Moderate drift
        PSI > 0.2: Significant drift (retrain!)
        """
        # Baseline ve current distribution'ları bin'lere böl
        bins = np.percentile(
            self.baseline_errors,
            np.linspace(0, 100, n_bins + 1)
        )
        
        # Baseline distribution
        baseline_counts, _ = np.histogram(self.baseline_errors, bins=bins)
        baseline_pct = baseline_counts / len(self.baseline_errors)
        baseline_pct[baseline_pct == 0] = 0.0001  # Avoid log(0)
        
        # Current distribution
        current_counts, _ = np.histogram(current_errors, bins=bins)
        current_pct = current_counts / len(current_errors)
        current_pct[current_pct == 0] = 0.0001
        
        # PSI calculation
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
        
        drift_detected = psi > self.psi_threshold
        
        if drift_detected:
            severity = 'CRITICAL' if psi > 0.3 else 'WARNING'
        else:
            severity = 'OK'
        
        result = {
            'method': 'PSI',
            'psi_value': psi,
            'threshold': self.psi_threshold,
            'drift_detected': drift_detected,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        }
        
        self.drift_history.append(result)
        
        return result
    
    def monitor_performance(self, y_true, y_pred, scores, 
                           metrics=None):
        """
        Performans metriklerini monitör et
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, 
            recall_score, f1_score, roc_auc_score
        )
        
        if metrics is None:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
            }
            
            if len(np.unique(y_true)) > 1:
                metrics['roc_auc'] = roc_auc_score(y_true, scores)
        
        monitoring_record = {
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        self.monitoring_data.append(monitoring_record)
        
        return monitoring_record
    
    def detect_performance_degradation(self, baseline_f1, 
                                      degradation_threshold=0.1):
        """
        Performans düşüşü tespit et
        """
        if len(self.monitoring_data) == 0:
            return None
        
        recent_f1 = self.monitoring_data[-1]['f1']
        degradation = baseline_f1 - recent_f1
        
        degraded = degradation > degradation_threshold
        
        return {
            'baseline_f1': baseline_f1,
            'current_f1': recent_f1,
            'degradation': degradation,
            'threshold': degradation_threshold,
            'degraded': degraded,
            'timestamp': datetime.now().isoformat()
        }
    
    def comprehensive_drift_check(self, current_errors, 
                                  y_true=None, y_pred=None, scores=None):
        """
        Tüm drift check'leri yap
        """
        results = {}
        
        # 1. KS test
        results['ks_test'] = self.detect_ks_drift(current_errors)
        
        # 2. PSI
        results['psi'] = self.calculate_psi(current_errors)
        
        # 3. Performance monitoring (eğer ground truth varsa)
        if y_true is not None and y_pred is not None:
            results['performance'] = self.monitor_performance(
                y_true, y_pred, scores
            )
        
        # Genel karar
        drift_detected = (
            results['ks_test']['drift_detected'] or
            results['psi']['drift_detected']
        )
        
        results['summary'] = {
            'drift_detected': drift_detected,
            'timestamp': datetime.now().isoformat(),
            'recommendation': 'RETRAIN MODEL' if drift_detected else 'OK'
        }
        
        return results
    
    def plot_drift_monitoring(self, save_path='drift_monitoring.png'):
        """
        Drift monitoring visualizasyonu
        """
        if len(self.drift_history) == 0:
            print("⚠️ No drift history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. KS test p-values over time
        ax = axes[0, 0]
        ks_records = [r for r in self.drift_history if r['method'] == 'KS-test']
        if ks_records:
            timestamps = [r['timestamp'] for r in ks_records]
            p_values = [r['p_value'] for r in ks_records]
            
            ax.plot(range(len(p_values)), p_values, 'b-o', linewidth=2)
            ax.axhline(self.ks_threshold, color='r', linestyle='--',
                      label=f'Threshold ({self.ks_threshold})')
            ax.set_xlabel('Time Window')
            ax.set_ylabel('P-Value')
            ax.set_title('KS Test - Drift Detection')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. PSI values over time
        ax = axes[0, 1]
        psi_records = [r for r in self.drift_history if r['method'] == 'PSI']
        if psi_records:
            psi_values = [r['psi_value'] for r in psi_records]
            
            ax.plot(range(len(psi_values)), psi_values, 'g-o', linewidth=2)
            ax.axhline(self.psi_threshold, color='r', linestyle='--',
                      label=f'Threshold ({self.psi_threshold})')
            ax.axhline(0.3, color='orange', linestyle='--',
                      label='Critical (0.3)')
            ax.set_xlabel('Time Window')
            ax.set_ylabel('PSI Value')
            ax.set_title('Population Stability Index')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. Performance metrics over time
        ax = axes[1, 0]
        if self.monitoring_data:
            df_monitor = pd.DataFrame(self.monitoring_data)
            
            if 'f1' in df_monitor.columns:
                ax.plot(df_monitor.index, df_monitor['f1'], 
                       'b-o', label='F1-Score', linewidth=2)
            if 'precision' in df_monitor.columns:
                ax.plot(df_monitor.index, df_monitor['precision'],
                       'g-s', label='Precision', linewidth=2)
            if 'recall' in df_monitor.columns:
                ax.plot(df_monitor.index, df_monitor['recall'],
                       'r-^', label='Recall', linewidth=2)
            
            ax.set_xlabel('Time Window')
            ax.set_ylabel('Score')
            ax.set_title('Performance Metrics Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Drift alerts summary
        ax = axes[1, 1]
        drift_alerts = [r for r in self.drift_history if r.get('drift_detected', False)]
        
        if drift_alerts:
            methods = [r['method'] for r in drift_alerts]
            from collections import Counter
            method_counts = Counter(methods)
            
            ax.bar(method_counts.keys(), method_counts.values(),
                  color=['#e74c3c', '#3498db'])
            ax.set_ylabel('Alert Count')
            ax.set_title('Drift Alerts by Method')
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Drift Detected ✅',
                   ha='center', va='center', fontsize=20,
                   color='green', fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Drift monitoring plot saved: {save_path}")
        
        return fig
    
    def export_report(self, output_file='drift_report.json'):
        """
        Drift raporu oluştur
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'baseline_stats': {
                'mean': float(np.mean(self.baseline_errors)),
                'std': float(np.std(self.baseline_errors)),
                'median': float(np.median(self.baseline_errors)),
                'count': len(self.baseline_errors)
            },
            'drift_history': self.drift_history,
            'monitoring_data': self.monitoring_data,
            'summary': {
                'total_checks': len(self.drift_history),
                'drift_alerts': sum(1 for r in self.drift_history 
                                   if r.get('drift_detected', False)),
                'last_check': self.drift_history[-1] if self.drift_history else None
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Drift report saved: {output_file}")
        
        return report


# =============================================================================
# KULLANIM
# =============================================================================

if __name__ == "__main__":
    
    # Simülasyon: İlk eğitim errors
    baseline_errors = np.random.gamma(2, 0.001, 5000)
    
    # Drift detector oluştur
    detector = ModelDriftDetector(
        baseline_errors,
        window_size=1000,
        ks_threshold=0.05,
        psi_threshold=0.2
    )
    
    print("\n" + "="*60)
    print("MODEL DRIFT DETECTION SIMULATION")
    print("="*60)
    
    # Simülasyon: 10 time window
    for i in range(10):
        print(f"\n--- Time Window {i+1} ---")
        
        # İlk 5 window: No drift
        if i < 5:
            current_errors = np.random.gamma(2, 0.001, 1000)
        # Son 5 window: Drift var!
        else:
            current_errors = np.random.gamma(2, 0.0015, 1000)  # Mean shifted!
        
        # Drift check
        results = detector.comprehensive_drift_check(current_errors)
        
        print(f"KS Test: p-value={results['ks_test']['p_value']:.4f}, "
              f"Drift={results['ks_test']['drift_detected']}")
        print(f"PSI: value={results['psi']['psi_value']:.4f}, "
              f"Drift={results['psi']['drift_detected']}")
        print(f"Recommendation: {results['summary']['recommendation']}")
    
    # Görselleştir
    detector.plot_drift_monitoring()
    
    # Rapor
    detector.export_report()
    
    print("\n✅ Drift detection simulation completed!")