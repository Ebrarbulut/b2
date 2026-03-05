"""
🎯 GERÇEK SALDIRILARI YAKALAMA VE ETİKETLEME SİSTEMİ
====================================================

Bu modül, kontrollü ortamda gerçek saldırı trafiği yakalayıp
etiketlemek için kullanılır.

KULLANIM:
1. Test ağı kur (VirtualBox/VMware)
2. Bu script'i çalıştır
3. Saldırı simülasyonlarını yap
4. Traffic'i otomatik etiketle
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import subprocess
import time
import os

class RealAttackTrafficGenerator:
    """
    Gerçek saldırı trafiği üretme ve etiketleme sistemi
    """
    
    def __init__(self, output_dir='real_traffic_data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.attack_log = []
        
    def log_attack(self, attack_type, source_ip, target_ip, 
                   start_time, end_time, command, description):
        """
        Saldırı bilgilerini kaydet
        """
        self.attack_log.append({
            'attack_type': attack_type,
            'source_ip': source_ip,
            'target_ip': target_ip,
            'start_time': start_time,
            'end_time': end_time,
            'command': command,
            'description': description,
            'duration_seconds': (end_time - start_time).total_seconds()
        })
        
        print(f"✅ Logged: {attack_type} - {start_time} to {end_time}")
    
    def save_attack_log(self):
        """
        Saldırı loglarını CSV'ye kaydet
        """
        df = pd.DataFrame(self.attack_log)
        log_file = os.path.join(self.output_dir, 'attack_ground_truth.csv')
        df.to_csv(log_file, index=False)
        print(f"\n📊 Attack log saved: {log_file}")
        print(f"Total attacks logged: {len(self.attack_log)}")
        return log_file
    
    def generate_attack_scenario(self, target_ip):
        """
        Önceden tanımlı saldırı senaryolarını üret
        
        NOT: Bu fonksiyonlar ÖRNEK amaçlıdır!
        Gerçek saldırı komutlarını SADECE test ağınızda çalıştırın!
        """
        scenarios = []
        
        # 1. PORT SCAN
        scenarios.append({
            'name': 'port_scan',
            'command': f'nmap -sS -T4 -p 1-1000 {target_ip}',
            'duration': 120,  # saniye
            'description': 'TCP SYN scan on ports 1-1000'
        })
        
        # 2. UDP SCAN
        scenarios.append({
            'name': 'udp_scan',
            'command': f'nmap -sU -T4 -p 53,161,500 {target_ip}',
            'duration': 90,
            'description': 'UDP scan on common ports'
        })
        
        # 3. SYN FLOOD (DDoS)
        scenarios.append({
            'name': 'syn_flood',
            'command': f'hping3 -S --flood -V -p 80 {target_ip}',
            'duration': 30,
            'description': 'SYN flood attack on port 80'
        })
        
        # 4. HTTP FLOOD
        scenarios.append({
            'name': 'http_flood',
            'command': f'ab -n 5000 -c 100 http://{target_ip}/',
            'duration': 60,
            'description': 'HTTP flood using Apache Bench'
        })
        
        # 5. SLOW SCAN (Stealthy)
        scenarios.append({
            'name': 'slow_scan',
            'command': f'nmap -sS -T1 -p 22,80,443 {target_ip}',
            'duration': 300,
            'description': 'Slow stealth scan (T1 timing)'
        })
        
        # 6. SERVICE DETECTION
        scenarios.append({
            'name': 'service_detection',
            'command': f'nmap -sV -p 80,443,22,21,3306 {target_ip}',
            'duration': 90,
            'description': 'Version detection scan'
        })
        
        return scenarios


class GroundTruthMatcher:
    """
    Zeek conn.log dosyasını saldırı loglarıyla eşleştirerek etiketle
    """
    
    def __init__(self, conn_log_path, attack_log_path):
        self.conn_log_path = conn_log_path
        self.attack_log_path = attack_log_path
        
    def load_conn_log(self):
        """
        Zeek conn.log'u yükle
        """
        # Zeek log headers
        column_names = [
            'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p',
            'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes',
            'conn_state', 'local_orig', 'local_resp', 'missed_bytes',
            'history', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes',
            'tunnel_parents'
        ]
        
        # Tab-separated, skip comment lines
        df = pd.read_csv(
            self.conn_log_path,
            sep='\t',
            comment='#',
            names=column_names,
            on_bad_lines='skip'
        )
        
        # Timestamp'i datetime'a çevir
        df['ts'] = pd.to_datetime(df['ts'], unit='s')
        
        print(f"✅ Loaded {len(df)} connections from conn.log")
        return df
    
    def load_attack_log(self):
        """
        Saldırı ground truth'u yükle
        """
        df = pd.read_csv(self.attack_log_path)
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
        
        print(f"✅ Loaded {len(df)} attack records")
        return df
    
    def match_labels(self, conn_df, attack_df):
        """
        Bağlantıları saldırı zamanlarıyla eşleştir
        """
        # Başlangıçta hepsi normal
        conn_df['label'] = 'normal'
        conn_df['attack_type'] = 'normal'
        
        # Her saldırı için
        for idx, attack in attack_df.iterrows():
            # Zaman aralığındaki bağlantıları bul
            time_mask = (
                (conn_df['ts'] >= attack['start_time']) &
                (conn_df['ts'] <= attack['end_time'])
            )
            
            # IP eşleşmesi (eğer varsa)
            if 'source_ip' in attack and pd.notna(attack['source_ip']):
                ip_mask = (
                    (conn_df['id.orig_h'] == attack['source_ip']) |
                    (conn_df['id.resp_h'] == attack['target_ip'])
                )
                final_mask = time_mask & ip_mask
            else:
                final_mask = time_mask
            
            # Etiketle
            conn_df.loc[final_mask, 'label'] = 'anomaly'
            conn_df.loc[final_mask, 'attack_type'] = attack['attack_type']
            
            matched_count = final_mask.sum()
            print(f"  {attack['attack_type']}: {matched_count} connections matched")
        
        return conn_df
    
    def create_labeled_dataset(self, output_path='labeled_traffic.csv'):
        """
        Etiketli veri seti oluştur
        """
        print("\n🔄 Creating labeled dataset...")
        
        # Verileri yükle
        conn_df = self.load_conn_log()
        attack_df = self.load_attack_log()
        
        # Etiketle
        labeled_df = self.match_labels(conn_df, attack_df)
        
        # İstatistikler
        print("\n📊 Label Distribution:")
        print(labeled_df['label'].value_counts())
        print("\n📊 Attack Type Distribution:")
        print(labeled_df['attack_type'].value_counts())
        
        # Kaydet
        labeled_df.to_csv(output_path, index=False)
        print(f"\n✅ Labeled dataset saved: {output_path}")
        
        return labeled_df


class AttackScenarioRunner:
    """
    Saldırı senaryolarını otomatik çalıştır ve logla
    
    ⚠️ SADECE TEST AĞINDA KULLANIN!
    """
    
    def __init__(self, traffic_generator):
        self.generator = traffic_generator
        self.pcap_file = None
        
    def start_packet_capture(self, interface='eth0', output_file='attack_traffic.pcap'):
        """
        Wireshark/tcpdump ile paket yakalamayı başlat
        """
        self.pcap_file = os.path.join(self.generator.output_dir, output_file)
        
        print(f"\n🎥 Starting packet capture on {interface}...")
        print(f"📁 Output: {self.pcap_file}")
        print("⚠️  Press Ctrl+C to stop capture\n")
        
        # tcpdump komutu (background'da çalışacak)
        cmd = f"sudo tcpdump -i {interface} -w {self.pcap_file}"
        
        # Subprocess olarak başlat
        process = subprocess.Popen(
            cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        time.sleep(2)  # tcpdump'ın başlamasını bekle
        print("✅ Packet capture started!")
        
        return process
    
    def run_attack_safely(self, attack_scenario, source_ip, target_ip):
        """
        Bir saldırı senaryosunu güvenli şekilde çalıştır
        """
        print(f"\n🚨 Running: {attack_scenario['name']}")
        print(f"   Command: {attack_scenario['command']}")
        print(f"   Duration: {attack_scenario['duration']}s")
        
        start_time = datetime.now()
        
        # Komutu çalıştır (timeout ile)
        try:
            subprocess.run(
                attack_scenario['command'].split(),
                timeout=attack_scenario['duration'],
                capture_output=True
            )
        except subprocess.TimeoutExpired:
            pass  # Normal, duration sonrası timeout bekleniyor
        except Exception as e:
            print(f"❌ Error: {e}")
        
        end_time = datetime.now()
        
        # Loga kaydet
        self.generator.log_attack(
            attack_type=attack_scenario['name'],
            source_ip=source_ip,
            target_ip=target_ip,
            start_time=start_time,
            end_time=end_time,
            command=attack_scenario['command'],
            description=attack_scenario['description']
        )
        
        print(f"✅ Completed: {attack_scenario['name']}")
        
    def run_full_scenario(self, source_ip, target_ip, 
                         interface='eth0', normal_traffic_duration=300):
        """
        Tam senaryo: Normal trafik + Saldırılar + Normal trafik
        """
        print("=" * 60)
        print("🎯 STARTING FULL ATTACK SCENARIO")
        print("=" * 60)
        
        # 1. Packet capture başlat
        capture_process = self.start_packet_capture(interface)
        
        try:
            # 2. İlk normal trafik (baseline)
            print(f"\n⏳ Capturing normal traffic for {normal_traffic_duration}s...")
            time.sleep(normal_traffic_duration)
            
            # 3. Saldırı senaryolarını çalıştır
            scenarios = self.generator.generate_attack_scenario(target_ip)
            
            for scenario in scenarios:
                # Her saldırı arasında 30 saniye normal trafik
                print("\n⏳ Normal traffic interval (30s)...")
                time.sleep(30)
                
                # Saldırıyı çalıştır
                self.run_attack_safely(scenario, source_ip, target_ip)
            
            # 4. Son normal trafik
            print(f"\n⏳ Final normal traffic ({normal_traffic_duration}s)...")
            time.sleep(normal_traffic_duration)
            
        finally:
            # 5. Packet capture'ı durdur
            print("\n🛑 Stopping packet capture...")
            capture_process.terminate()
            time.sleep(2)
            
            # 6. Zeek ile işle
            print("\n🔄 Processing with Zeek...")
            self.process_with_zeek()
            
            # 7. Attack log'u kaydet
            self.generator.save_attack_log()
            
        print("\n" + "=" * 60)
        print("✅ SCENARIO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
    
    def process_with_zeek(self):
        """
        PCAP dosyasını Zeek ile işle
        """
        if not self.pcap_file:
            print("❌ No PCAP file to process")
            return
        
        zeek_output = os.path.join(self.generator.output_dir, 'zeek_logs')
        os.makedirs(zeek_output, exist_ok=True)
        
        cmd = f"zeek -r {self.pcap_file} -C"
        
        print(f"Running: {cmd}")
        subprocess.run(cmd.split(), cwd=zeek_output)
        
        print(f"✅ Zeek processing complete: {zeek_output}/conn.log")


# =============================================================================
# KULLANIM ÖRNEĞİ
# =============================================================================

if __name__ == "__main__":
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  REAL ATTACK TRAFFIC GENERATOR & GROUND TRUTH CREATOR      ║
    ║                                                            ║
    ║  ⚠️  SADECE TEST AĞINDA KULLANIN!                         ║
    ║  ⚠️  YASAL OLARAK İZİN VERİLEN ORTAMLARDA ÇALIŞTIRIN!    ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Kullanım modu seç
    print("\nSeçenekler:")
    print("1. Tam senaryo çalıştır (Wireshark + Saldırılar + Zeek)")
    print("2. Sadece mevcut conn.log'u etiketle")
    print("3. Manuel saldırı loglama (interactive)")
    
    choice = input("\nSeçiminiz (1-3): ")
    
    if choice == "1":
        # Tam senaryo
        source_ip = input("Source IP (kendi IP'niz): ")
        target_ip = input("Target IP (test VM IP'si): ")
        interface = input("Network interface (default: eth0): ") or "eth0"
        
        generator = RealAttackTrafficGenerator()
        runner = AttackScenarioRunner(generator)
        
        runner.run_full_scenario(
            source_ip=source_ip,
            target_ip=target_ip,
            interface=interface,
            normal_traffic_duration=300  # 5 dakika
        )
        
    elif choice == "2":
        # Sadece etiketleme
        conn_log = input("conn.log path: ")
        attack_log = input("attack_ground_truth.csv path: ")
        
        matcher = GroundTruthMatcher(conn_log, attack_log)
        labeled_df = matcher.create_labeled_dataset()
        
        print("\n✅ Etiketli veri hazır! Model testine geçebilirsiniz.")
        
    elif choice == "3":
        # Manuel loglama
        generator = RealAttackTrafficGenerator()
        
        print("\n📝 Manuel saldırı loglama modu")
        print("Her saldırıdan ÖNCE ve SONRA buraya gelin!")
        
        while True:
            print("\n" + "="*50)
            action = input("\nAction (start/end/save/quit): ").lower()
            
            if action == "start":
                attack_type = input("Attack type (e.g., port_scan): ")
                source_ip = input("Source IP: ")
                target_ip = input("Target IP: ")
                command = input("Command used: ")
                
                start_time = datetime.now()
                print(f"\n⏰ Start time logged: {start_time}")
                print("🚨 NOW RUN YOUR ATTACK!")
                input("Press Enter when attack is FINISHED...")
                
                end_time = datetime.now()
                
                generator.log_attack(
                    attack_type=attack_type,
                    source_ip=source_ip,
                    target_ip=target_ip,
                    start_time=start_time,
                    end_time=end_time,
                    command=command,
                    description=input("Description: ")
                )
                
            elif action == "save":
                generator.save_attack_log()
                
            elif action == "quit":
                generator.save_attack_log()
                break
    
    print("\n🎉 İşlem tamamlandı!")
