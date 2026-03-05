"""
🔍 EXPLAINABLE ANOMALY DETECTION
=================================

Bu modül, tespit edilen anomalileri açıklar ve SOC analisti için
actionable alert'ler üretir.

ÖZELLİKLER:
- Feature-level contribution analysis
- Severity scoring
- Recommended actions
- Contextual information
- Alert prioritization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

class AnomalyExplainer:
    """
    Anomali açıklama ve önceliklendirme sistemi
    """
    
    def __init__(self, feature_names, threshold):
        """
        Args:
            feature_names: Feature isimleri listesi
            threshold: Anomali threshold değeri
        """
        self.feature_names = feature_names
        self.threshold = threshold
        
        # Feature bazlı action mapping
        self.action_map = self._create_action_map()
        
        # Severity thresholds
        self.severity_levels = {
            'LOW': (1.0, 1.5),      # 1x - 1.5x threshold
            'MEDIUM': (1.5, 3.0),   # 1.5x - 3x threshold
            'HIGH': (3.0, 5.0),     # 3x - 5x threshold
            'CRITICAL': (5.0, float('inf'))  # 5x+ threshold
        }
    
    def _create_action_map(self):
        """
        Her feature için recommended action tanımla
        """
        return {
            'duration': {
                'description': 'Uzun süreli bağlantı tespit edildi',
                'actions': [
                    'Session timeout kontrolü yapın',
                    'Long-term connection log\'larını inceleyin',
                    'C&C communication olasılığını değerlendirin'
                ],
                'indicators': ['Persistent backdoor', 'Data exfiltration', 'Tunnel']
            },
            'orig_bytes': {
                'description': 'Anormal yüksek veri gönderimi',
                'actions': [
                    'Data exfiltration araştırması yapın',
                    'Gönderilen dosya/veri tiplerini kontrol edin',
                    'Hedefe giden trafik içeriğini analiz edin'
                ],
                'indicators': ['Data theft', 'Database dump', 'File upload']
            },
            'resp_bytes': {
                'description': 'Anormal yüksek veri indirimi',
                'actions': [
                    'Malware indirme olasılığını kontrol edin',
                    'Payload analizi yapın',
                    'C&C server iletişimini araştırın'
                ],
                'indicators': ['Malware download', 'C&C communication', 'Exploit payload']
            },
            'orig_pkts': {
                'description': 'Fazla paket gönderimi',
                'actions': [
                    'Port scan aktivitesi kontrol edin',
                    'DDoS saldırısı olasılığını değerlendirin',
                    'Packet içeriklerini inceleyin'
                ],
                'indicators': ['Port scanning', 'Network mapping', 'Flood attack']
            },
            'resp_pkts': {
                'description': 'Fazla paket alımı',
                'actions': [
                    'DDoS reflection olasılığını kontrol edin',
                    'Service enumeration araştırın',
                    'Response pattern\'ini analiz edin'
                ],
                'indicators': ['Service scan', 'DDoS reflection', 'Amplification']
            },
            'bytes_ratio': {
                'description': 'Anormal upload/download oranı',
                'actions': [
                    'Trafik akışını analiz edin',
                    'Upload/Download dengesizliğini inceleyin',
                    'Olası backdoor iletişimini araştırın'
                ],
                'indicators': ['Asymmetric communication', 'Covert channel']
            },
            'throughput': {
                'description': 'Anormal veri transfer hızı',
                'actions': [
                    'Bandwidth usage\'ı monitör edin',
                    'Bulk data transfer\'i araştırın',
                    'Rate limiting uygulamayı düşünün'
                ],
                'indicators': ['Bulk transfer', 'DDoS', 'Bandwidth abuse']
            },
            'proto': {
                'description': 'Beklenmeyen protokol kullanımı',
                'actions': [
                    'Protokol policy\'lerini kontrol edin',
                    'Unauthorized protocol usage araştırın',
                    'Firewall rule\'larını gözden geçirin'
                ],
                'indicators': ['Protocol violation', 'Tunneling', 'Covert channel']
            },
            'service': {
                'description': 'Anormal servis aktivitesi',
                'actions': [
                    'Servis erişim log\'larını inceleyin',
                    'Unauthorized service access kontrol edin',
                    'Service fingerprinting araştırın'
                ],
                'indicators': ['Service abuse', 'Exploitation attempt']
            },
            'conn_state': {
                'description': 'Beklenmeyen bağlantı durumu',
                'actions': [
                    'Connection state pattern\'ini analiz edin',
                    'Failed connection\'ları araştırın',
                    'Reset/timeout sebeplerini kontrol edin'
                ],
                'indicators': ['Connection tampering', 'Scan activity', 'DoS']
            }
        }
    
    def explain_single_anomaly(self, original, reconstructed, 
                              connection_info=None):
        """
        Tek bir anomaliyi detaylı açıkla
        
        Args:
            original: Orijinal feature değerleri
            reconstructed: Reconstruct edilmiş değerler
            connection_info: Ek bağlantı bilgileri (IP, port, vb.)
        
        Returns:
            dict: Açıklama bilgileri
        """
        # Feature-level errors
        feature_errors = np.abs(original - reconstructed)
        total_error = np.mean(feature_errors)
        
        # Feature contributions (yüzde olarak)
        contributions = (feature_errors / total_error) * 100
        
        # En yüksek 3 anomalous feature
        top_indices = np.argsort(contributions)[-3:][::-1]
        
        # Severity hesapla
        severity = self._calculate_severity(total_error)
        
        # Priority score (0-100)
        priority = self._calculate_priority_score(
            total_error, severity, feature_errors
        )
        
        # Feature detayları
        anomalous_features = []
        for idx in top_indices:
            feature_name = self.feature_names[idx]
            
            feature_detail = {
                'name': feature_name,
                'contribution_percent': float(contributions[idx]),
                'original_value': float(original[idx]),
                'expected_value': float(reconstructed[idx]),
                'deviation': float(feature_errors[idx]),
                'action_info': self.action_map.get(
                    feature_name, 
                    {'description': 'Unknown feature', 'actions': [], 'indicators': []}
                )
            }
            
            anomalous_features.append(feature_detail)
        
        # Recommended actions
        recommended_actions = self._generate_actions(anomalous_features)
        
        # Explanation oluştur
        explanation = {
            'timestamp': datetime.now().isoformat(),
            'anomaly_score': float(total_error),
            'threshold': float(self.threshold),
            'severity': severity,
            'priority_score': priority,
            'anomalous_features': anomalous_features,
            'recommended_actions': recommended_actions,
            'investigation_steps': self._generate_investigation_steps(anomalous_features),
            'potential_threats': self._identify_threats(anomalous_features),
            'connection_info': connection_info or {}
        }
        
        return explanation
    
    def _calculate_severity(self, error):
        """
        Severity level hesapla
        """
        ratio = error / self.threshold
        
        for level, (min_ratio, max_ratio) in self.severity_levels.items():
            if min_ratio <= ratio < max_ratio:
                return level
        
        return 'UNKNOWN'
    
    def _calculate_priority_score(self, error, severity, feature_errors):
        """
        0-100 arası priority score
        
        Factors:
        - Anomaly magnitude (얼마 sapıyor?)
        - Feature diversity (kaç feature anormal?)
        - Severity level
        """
        # Base score: Error magnitude (0-50)
        ratio = min(error / self.threshold, 10)  # Cap at 10x
        magnitude_score = (ratio / 10) * 50
        
        # Diversity score: Kaç feature ciddi şekilde sapıyor? (0-30)
        significant_deviations = np.sum(
            feature_errors > (np.mean(feature_errors) + np.std(feature_errors))
        )
        diversity_score = min(significant_deviations / len(feature_errors), 1) * 30
        
        # Severity bonus (0-20)
        severity_bonus = {
            'LOW': 5,
            'MEDIUM': 10,
            'HIGH': 15,
            'CRITICAL': 20
        }.get(severity, 0)
        
        total_score = magnitude_score + diversity_score + severity_bonus
        
        return min(int(total_score), 100)
    
    def _generate_actions(self, anomalous_features):
        """
        Recommended action'ları derle
        """
        all_actions = []
        seen_actions = set()
        
        for feature in anomalous_features:
            actions = feature['action_info']['actions']
            for action in actions:
                if action not in seen_actions:
                    all_actions.append({
                        'action': action,
                        'related_feature': feature['name'],
                        'priority': 'HIGH' if feature == anomalous_features[0] else 'MEDIUM'
                    })
                    seen_actions.add(action)
        
        return all_actions
    
    def _generate_investigation_steps(self, anomalous_features):
        """
        Investigation checklist üret
        """
        steps = [
            {
                'step': 1,
                'action': 'Verify the alert is not a false positive',
                'details': 'Check if similar behavior exists in historical data'
            },
            {
                'step': 2,
                'action': 'Identify the source',
                'details': 'Determine if source IP is internal/external, known/unknown'
            },
            {
                'step': 3,
                'action': 'Analyze traffic content',
                'details': 'Inspect payload if possible, check for malicious patterns'
            }
        ]
        
        # Feature-specific steps
        step_num = 4
        for feature in anomalous_features[:2]:  # Top 2 features
            steps.append({
                'step': step_num,
                'action': f"Investigate {feature['name']} anomaly",
                'details': feature['action_info']['description']
            })
            step_num += 1
        
        steps.append({
            'step': step_num,
            'action': 'Document findings and take action',
            'details': 'Block if malicious, whitelist if benign, escalate if unsure'
        })
        
        return steps
    
    def _identify_threats(self, anomalous_features):
        """
        Olası threat'leri tanımla
        """
        all_threats = []
        
        for feature in anomalous_features:
            threats = feature['action_info']['indicators']
            all_threats.extend(threats)
        
        # Deduplicate
        unique_threats = list(set(all_threats))
        
        return unique_threats
    
    def generate_alert_report(self, explanations, output_file='alert_report.json'):
        """
        Tüm anomaliler için comprehensive report
        """
        # Priority'ye göre sırala
        sorted_explanations = sorted(
            explanations,
            key=lambda x: x['priority_score'],
            reverse=True
        )
        
        # Summary statistics
        summary = {
            'total_anomalies': len(explanations),
            'severity_distribution': {},
            'average_priority': np.mean([e['priority_score'] for e in explanations]),
            'top_anomalous_features': {},
            'common_threats': {}
        }
        
        # Severity distribution
        for exp in explanations:
            severity = exp['severity']
            summary['severity_distribution'][severity] = \
                summary['severity_distribution'].get(severity, 0) + 1
        
        # Top features
        feature_counts = {}
        for exp in explanations:
            for feat in exp['anomalous_features']:
                name = feat['name']
                feature_counts[name] = feature_counts.get(name, 0) + 1
        
        summary['top_anomalous_features'] = dict(
            sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        )
        
        # Common threats
        threat_counts = {}
        for exp in explanations:
            for threat in exp['potential_threats']:
                threat_counts[threat] = threat_counts.get(threat, 0) + 1
        
        summary['common_threats'] = dict(
            sorted(threat_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        
        # Full report
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': summary,
            'alerts': sorted_explanations
        }
        
        # Save
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Alert report saved: {output_file}")
        
        return report
    
    def create_alert_dashboard_data(self, explanations):
        """
        Streamlit dashboard için veri hazırla
        """
        df = pd.DataFrame([
            {
                'timestamp': exp['timestamp'],
                'severity': exp['severity'],
                'priority': exp['priority_score'],
                'anomaly_score': exp['anomaly_score'],
                'top_feature': exp['anomalous_features'][0]['name'],
                'top_threat': exp['potential_threats'][0] if exp['potential_threats'] else 'Unknown'
            }
            for exp in explanations
        ])
        
        return df
    
    def visualize_feature_contributions(self, explanation, save_path=None):
        """
        Bir anomali için feature contribution chart
        """
        features = explanation['anomalous_features']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        names = [f['name'] for f in features]
        contributions = [f['contribution_percent'] for f in features]
        colors = ['#e74c3c' if i == 0 else '#3498db' for i in range(len(names))]
        
        ax.barh(names, contributions, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Contribution (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Anomaly Explanation - Severity: {explanation["severity"]} '
            f'(Priority: {explanation["priority_score"]}/100)',
            fontsize=14,
            fontweight='bold'
        )
        ax.grid(axis='x', alpha=0.3)
        
        # Annotations
        for i, (name, contrib) in enumerate(zip(names, contributions)):
            ax.text(
                contrib + 1, i,
                f'{contrib:.1f}%',
                va='center',
                fontsize=10,
                fontweight='bold'
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Feature contribution plot saved: {save_path}")
        
        return fig


class AlertPrioritizer:
    """
    Alert'leri önceliklendirme ve filtreleme sistemi
    """
    
    def __init__(self, min_priority=50, max_alerts_per_hour=100):
        """
        Args:
            min_priority: Minimum priority score (altındakiler filtrelenir)
            max_alerts_per_hour: Saat başına max alert sayısı
        """
        self.min_priority = min_priority
        self.max_alerts_per_hour = max_alerts_per_hour
        self.alert_history = []
    
    def filter_alerts(self, explanations):
        """
        Alert'leri priority'ye göre filtrele
        """
        # Priority threshold
        filtered = [
            exp for exp in explanations
            if exp['priority_score'] >= self.min_priority
        ]
        
        # Rate limiting (son 1 saat)
        recent_alerts = [
            a for a in self.alert_history
            if (datetime.now() - datetime.fromisoformat(a['timestamp'])).seconds < 3600
        ]
        
        if len(recent_alerts) >= self.max_alerts_per_hour:
            print(f"⚠️ Alert rate limit reached ({self.max_alerts_per_hour}/hour)")
            # Sadece en yüksek priority'li alert'leri al
            filtered = sorted(
                filtered,
                key=lambda x: x['priority_score'],
                reverse=True
            )[:10]
        
        # History'ye ekle
        self.alert_history.extend(filtered)
        
        return filtered
    
    def deduplicate_alerts(self, explanations, similarity_threshold=0.8):
        """
        Benzer alert'leri grupla (alert fatigue'i önle)
        """
        # Basit implementation: aynı top feature'a sahip alert'leri grupla
        grouped = {}
        
        for exp in explanations:
            top_feature = exp['anomalous_features'][0]['name']
            
            if top_feature not in grouped:
                grouped[top_feature] = []
            
            grouped[top_feature].append(exp)
        
        # Her gruptan en yüksek priority'li alert'i al
        deduplicated = []
        for feature, group in grouped.items():
            if len(group) > 1:
                print(f"   Grouped {len(group)} similar alerts (feature: {feature})")
            
            # En yüksek priority'li alert'i seç
            best_alert = max(group, key=lambda x: x['priority_score'])
            best_alert['similar_alerts_count'] = len(group)
            
            deduplicated.append(best_alert)
        
        return deduplicated


# =============================================================================
# STREAMLIT ENTEGRASYONU
# =============================================================================

def integrate_with_streamlit(explainer, anomalies_df, model, X_test):
    """
    Streamlit app'e entegrasyon örneği
    """
    import streamlit as st
    
    st.header("🔍 Anomaly Explanation & Investigation")
    
    # Priority filtreleme
    min_priority = st.slider("Minimum Priority", 0, 100, 50)
    
    # Her anomali için explanation
    explanations = []
    
    for idx in anomalies_df[anomalies_df['is_anomaly'] == 1].index:
        original = X_test[idx]
        reconstructed = model.predict(original.reshape(1, -1))[0]
        
        conn_info = {
            'source_ip': anomalies_df.loc[idx, 'id.orig_h'],
            'dest_ip': anomalies_df.loc[idx, 'id.resp_h'],
            'timestamp': anomalies_df.loc[idx, 'ts']
        }
        
        exp = explainer.explain_single_anomaly(
            original, reconstructed, conn_info
        )
        
        if exp['priority_score'] >= min_priority:
            explanations.append(exp)
    
    # Priority'ye göre sırala
    explanations = sorted(
        explanations,
        key=lambda x: x['priority_score'],
        reverse=True
    )
    
    st.write(f"**Found {len(explanations)} high-priority anomalies**")
    
    # Her anomali için card
    for i, exp in enumerate(explanations[:10], 1):  # Top 10
        with st.expander(
            f"#{i} - {exp['severity']} - Priority: {exp['priority_score']}/100"
        ):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Anomaly Score", f"{exp['anomaly_score']:.6f}")
            with col2:
                st.metric("Severity", exp['severity'])
            with col3:
                st.metric("Priority", f"{exp['priority_score']}/100")
            
            st.markdown("**Top Anomalous Features:**")
            for feat in exp['anomalous_features']:
                st.write(f"- **{feat['name']}**: {feat['contribution_percent']:.1f}% "
                        f"(Value: {feat['original_value']:.2f}, Expected: {feat['expected_value']:.2f})")
            
            st.markdown("**🚨 Recommended Actions:**")
            for action in exp['recommended_actions']:
                st.warning(f"**{action['priority']}:** {action['action']}")
            
            st.markdown("**🔎 Investigation Steps:**")
            for step in exp['investigation_steps']:
                st.info(f"{step['step']}. {step['action']}")
            
            st.markdown("**⚠️ Potential Threats:**")
            st.write(", ".join(exp['potential_threats']))


# =============================================================================
# KULLANIM ÖRNEĞİ
# =============================================================================

if __name__ == "__main__":
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║          EXPLAINABLE ANOMALY DETECTION SYSTEM              ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Örnek veri
    feature_names = [
        'duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts',
        'bytes_ratio', 'pkts_ratio', 'throughput', 'proto', 'service',
        'conn_state', 'total_bytes'
    ]
    
    threshold = 0.001
    
    # Explainer oluştur
    explainer = AnomalyExplainer(feature_names, threshold)
    
    # Örnek anomali
    original = np.array([100, 50000, 1000, 500, 50, 50, 10, 500, 1, 2, 1, 51000])
    reconstructed = np.array([10, 5000, 5000, 50, 50, 1, 1, 50, 1, 2, 1, 10000])
    
    conn_info = {
        'source_ip': '192.168.1.100',
        'dest_ip': '8.8.8.8',
        'timestamp': datetime.now().isoformat()
    }
    
    # Açıklama üret
    explanation = explainer.explain_single_anomaly(
        original, reconstructed, conn_info
    )
    
    # Sonuçları göster
    print("\n📊 ANOMALY EXPLANATION:")
    print("=" * 60)
    print(json.dumps(explanation, indent=2))
    
    # Visualization
    explainer.visualize_feature_contributions(
        explanation,
        save_path='anomaly_explanation_example.png'
    )
    
    print("\n✅ Explainable anomaly detection demo completed!")
