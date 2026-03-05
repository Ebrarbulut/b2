"""
PCAP dosyasını basit conn.log formatına çevir (Zeek olmadan)
"""
import sys
from pathlib import Path

try:
    from scapy.all import rdpcap, IP, TCP, UDP
    from scapy.layers.inet import ICMP
except ImportError:
    print("❌ Scapy kurulu değil!")
    print("💡 Kurulum: pip install scapy")
    sys.exit(1)

def pcap_to_connlog(pcap_file, output_file="conn.log"):
    """PCAP'i basit conn.log formatına çevir"""
    
    print(f"📦 PCAP dosyası okunuyor: {pcap_file}")
    packets = rdpcap(pcap_file)
    print(f"✅ {len(packets)} paket bulundu")
    
    connections = []
    
    for pkt in packets:
        if IP in pkt:
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            
            # Port bilgileri
            src_port = 0
            dst_port = 0
            proto = "unknown"
            
            if TCP in pkt:
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
                proto = "tcp"
            elif UDP in pkt:
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport
                proto = "udp"
            elif ICMP in pkt:
                proto = "icmp"
            
            # Basit conn.log formatı
            conn = {
                'ts': float(pkt.time),
                'id.orig_h': src_ip,
                'id.orig_p': src_port,
                'id.resp_h': dst_ip,
                'id.resp_p': dst_port,
                'proto': proto,
                'duration': 0.0,
                'orig_bytes': len(pkt),
                'resp_bytes': 0,
                'conn_state': 'S0',
                'service': '-'
            }
            connections.append(conn)
    
    # conn.log formatında yaz
    print(f"📝 conn.log yazılıyor: {output_file}")
    
    with open(output_file, 'w') as f:
        # Header
        f.write("#separator \\t\n")
        f.write("#fields\tts\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tduration\torig_bytes\tresp_bytes\tconn_state\tservice\n")
        
        # Data
        for conn in connections:
            line = f"{conn['ts']}\t{conn['id.orig_h']}\t{conn['id.orig_p']}\t{conn['id.resp_h']}\t{conn['id.resp_p']}\t{conn['proto']}\t{conn['duration']}\t{conn['orig_bytes']}\t{conn['resp_bytes']}\t{conn['conn_state']}\t{conn['service']}\n"
            f.write(line)
    
    print(f"✅ {len(connections)} bağlantı yazıldı")
    print(f"📁 Dosya: {output_file}")
    return output_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python pcap_to_connlog_simple.py <pcap_dosyası>")
        print("Örnek: python pcap_to_connlog_simple.py normal_traffic.pcap")
        sys.exit(1)
    
    pcap_file = sys.argv[1]
    
    if not Path(pcap_file).exists():
        print(f"❌ Dosya bulunamadı: {pcap_file}")
        sys.exit(1)
    
    output_file = Path(pcap_file).stem + "_conn.log"
    pcap_to_connlog(pcap_file, output_file)
    print("\n✅ TAMAMLANDI!")
