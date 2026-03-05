"""
Advanced Threshold Optimization
================================

ROC curve, PR curve, F1-score, cost-sensitive optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    f1_score, precision_score, recall_score,
    confusion_matrix
)

class ThresholdOptimizer:
    """
    Multiple threshold optimization strategies
    """
    
    def __init__(self, y_true, scores):
        """
        Args:
            y_true: Ground truth labels (0=normal, 1=anomaly)
            scores: Anomaly scores (continuous)
        """
        self.y_true = y_true
        self.scores = scores
        
        # ROC curve hesapla
        self.fpr, self.tpr, self.roc_thresholds = roc_curve(y_true, scores)
        self.roc_auc = auc(self.fpr, self.tpr)
        
        # PR curve hesapla
        self.precision, self.recall, self.pr_thresholds = precision_recall_curve(
            y_true, scores
        )
    
    def optimize_youden(self):
        """
        Youden's J statistic (TPR - FPR maksimizasyonu)
        """
        j_scores = self.tpr - self.fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = self.roc_thresholds[optimal_idx]
        
        return {
            'method': 'Youden',
            'threshold': optimal_threshold,
            'tpr': self.tpr[optimal_idx],
            'fpr': self.fpr[optimal_idx],
            'j_score': j_scores[optimal_idx]
        }
    
    def optimize_f1(self, beta=1.0):
        """
        F-beta score maksimizasyonu
        
        Args:
            beta: F-beta'daki beta değeri (1.0 = F1, 2.0 = recall'a ağırlık, vb.)
        """
        # Farklı threshold'larda F-beta hesapla
        thresholds = np.percentile(self.scores, np.linspace(0, 100, 1000))
        
        best_fbeta = 0
        best_threshold = 0
        best_precision = 0
        best_recall = 0
        
        for threshold in thresholds:
            y_pred = (self.scores > threshold).astype(int)
            
            # Hiç pozitif prediction yoksa atla
            if y_pred.sum() == 0:
                continue
            
            precision = precision_score(self.y_true, y_pred, zero_division=0)
            recall = recall_score(self.y_true, y_pred, zero_division=0)
            
            if precision + recall == 0:
                continue
            
            fbeta = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)
            
            if fbeta > best_fbeta:
                best_fbeta = fbeta
                best_threshold = threshold
                best_precision = precision
                best_recall = recall
        
        return {
            'method': f'F{beta}-score',
            'threshold': best_threshold,
            'f_beta': best_fbeta,
            'precision': best_precision,
            'recall': best_recall
        }
    
    def optimize_cost_sensitive(self, fp_cost=1, fn_cost=10):
        """
        Cost-sensitive threshold
        
        Args:
            fp_cost: False positive maliyeti
            fn_cost: False negative maliyeti (genelde daha yüksek!)
        """
        thresholds = np.percentile(self.scores, np.linspace(0, 100, 1000))
        
        best_cost = float('inf')
        best_threshold = 0
        best_fp = 0
        best_fn = 0
        
        for threshold in thresholds:
            y_pred = (self.scores > threshold).astype(int)
            
            fp = ((y_pred == 1) & (self.y_true == 0)).sum()
            fn = ((y_pred == 0) & (self.y_true == 1)).sum()
            
            total_cost = fp * fp_cost + fn * fn_cost
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_threshold = threshold
                best_fp = fp
                best_fn = fn
        
        return {
            'method': 'Cost-sensitive',
            'threshold': best_threshold,
            'total_cost': best_cost,
            'fp_count': best_fp,
            'fn_count': best_fn,
            'fp_cost': fp_cost,
            'fn_cost': fn_cost
        }
    
    def optimize_precision_target(self, target_precision=0.95):
        """
        Belirli bir precision'ı garanti eden en yüksek recall threshold'u
        """
        # PR curve'den target precision'ı sağlayan threshold'ları bul
        valid_indices = self.precision >= target_precision
        
        if not np.any(valid_indices):
            print(f"⚠️ {target_precision} precision ulaşılamıyor!")
            return None
        
        # En yüksek recall'lı threshold
        valid_recalls = self.recall[valid_indices]
        valid_thresholds = self.pr_thresholds[valid_indices[:-1]]  # PR curve bir eleman eksik
        
        best_idx = np.argmax(valid_recalls)
        
        return {
            'method': f'Precision-{target_precision}',
            'threshold': valid_thresholds[best_idx],
            'precision': self.precision[valid_indices][best_idx],
            'recall': valid_recalls[best_idx]
        }
    
    def optimize_recall_target(self, target_recall=0.95):
        """
        Belirli bir recall'ı garanti eden en yüksek precision threshold'u
        """
        valid_indices = self.recall >= target_recall
        
        if not np.any(valid_indices):
            print(f"⚠️ {target_recall} recall ulaşılamıyor!")
            return None
        
        valid_precisions = self.precision[valid_indices]
        valid_thresholds = self.pr_thresholds[valid_indices[:-1]]
        
        best_idx = np.argmax(valid_precisions)
        
        return {
            'method': f'Recall-{target_recall}',
            'threshold': valid_thresholds[best_idx],
            'precision': valid_precisions[best_idx],
            'recall': self.recall[valid_indices][best_idx]
        }
    
    def compare_all_methods(self):
        """
        Tüm yöntemleri karşılaştır
        """
        results = []
        
        # Youden
        results.append(self.optimize_youden())
        
        # F1
        results.append(self.optimize_f1(beta=1.0))
        
        # F2 (recall'a ağırlık)
        results.append(self.optimize_f1(beta=2.0))
        
        # Cost-sensitive
        results.append(self.optimize_cost_sensitive(fp_cost=1, fn_cost=10))
        
        # Precision-target
        prec_result = self.optimize_precision_target(target_precision=0.90)
        if prec_result:
            results.append(prec_result)
        
        # Recall-target
        recall_result = self.optimize_recall_target(target_recall=0.90)
        if recall_result:
            results.append(recall_result)
        
        # Percentile (mevcut yöntem)
        percentile_threshold = np.percentile(self.scores, 99)
        y_pred_perc = (self.scores > percentile_threshold).astype(int)
        results.append({
            'method': 'Percentile-99',
            'threshold': percentile_threshold,
            'precision': precision_score(self.y_true, y_pred_perc, zero_division=0),
            'recall': recall_score(self.y_true, y_pred_perc, zero_division=0),
            'f1': f1_score(self.y_true, y_pred_perc, zero_division=0)
        })
        
        return pd.DataFrame(results)
    
    def plot_comparison(self, save_path='threshold_comparison.png'):
        """
        Görsel karşılaştırma
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. ROC Curve
        ax = axes[0, 0]
        ax.plot(self.fpr, self.tpr, 'b-', linewidth=2, 
                label=f'ROC (AUC={self.roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        
        # Youden point
        youden_result = self.optimize_youden()
        youden_idx = np.argmin(np.abs(self.roc_thresholds - youden_result['threshold']))
        ax.plot(self.fpr[youden_idx], self.tpr[youden_idx], 
                'ro', markersize=10, label='Youden')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curve
        ax = axes[0, 1]
        ax.plot(self.recall, self.precision, 'g-', linewidth=2)
        
        # F1 point
        f1_result = self.optimize_f1()
        f1_threshold = f1_result['threshold']
        y_pred_f1 = (self.scores > f1_threshold).astype(int)
        f1_precision = precision_score(self.y_true, y_pred_f1)
        f1_recall = recall_score(self.y_true, y_pred_f1)
        ax.plot(f1_recall, f1_precision, 'ro', markersize=10, label='F1-optimal')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. F1-Score vs Threshold
        ax = axes[1, 0]
        thresholds_plot = np.percentile(self.scores, np.linspace(80, 99.9, 100))
        f1_scores = []
        
        for threshold in thresholds_plot:
            y_pred = (self.scores > threshold).astype(int)
            if y_pred.sum() > 0:
                f1_scores.append(f1_score(self.y_true, y_pred, zero_division=0))
            else:
                f1_scores.append(0)
        
        ax.plot(thresholds_plot, f1_scores, 'b-', linewidth=2)
        ax.axvline(f1_result['threshold'], color='r', linestyle='--', 
                   label=f"Optimal={f1_result['threshold']:.6f}")
        ax.set_xlabel('Threshold')
        ax.set_ylabel('F1-Score')
        ax.set_title('F1-Score vs Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Method Comparison
        ax = axes[1, 1]
        comparison_df = self.compare_all_methods()
        
        # F1 skorlarını hesapla
        for idx, row in comparison_df.iterrows():
            if 'f1' not in row or pd.isna(row['f1']):
                threshold = row['threshold']
                y_pred = (self.scores > threshold).astype(int)
                comparison_df.loc[idx, 'f1'] = f1_score(self.y_true, y_pred, zero_division=0)
        
        methods = comparison_df['method'].values
        f1_values = comparison_df['f1'].values
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
        bars = ax.barh(methods, f1_values, color=colors)
        ax.set_xlabel('F1-Score')
        ax.set_title('Method Comparison (F1-Score)')
        ax.grid(axis='x', alpha=0.3)
        
        # Değerleri bar'ların üzerine yaz
        for bar, value in zip(bars, f1_values):
            ax.text(value + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comparison plot saved: {save_path}")
        
        return fig


# =============================================================================
# KULLANIM
# =============================================================================

if __name__ == "__main__":
    
    # Test sonuçlarını yükle
    results = pd.read_csv('real_test_results.csv')
    
    y_true = (results['label'] != 'normal').astype(int)
    scores = results['anomaly_score'].values
    
    # Optimizer
    optimizer = ThresholdOptimizer(y_true, scores)
    
    # Tüm yöntemleri karşılaştır
    print("\n" + "="*60)
    print("THRESHOLD OPTIMIZATION COMPARISON")
    print("="*60)
    
    comparison_df = optimizer.compare_all_methods()
    print("\n", comparison_df.to_string(index=False))
    
    # En iyi yöntemi seç
    best_method = comparison_df.loc[comparison_df['f1'].idxmax()]
    print(f"\n✅ BEST METHOD: {best_method['method']}")
    print(f"   Threshold: {best_method['threshold']:.6f}")
    print(f"   F1-Score: {best_method['f1']:.4f}")
    
    # Görselleştir
    optimizer.plot_comparison()
    
    # Kaydet
    comparison_df.to_csv('threshold_optimization_results.csv', index=False)