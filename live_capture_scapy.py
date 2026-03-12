from scapy.all import sniff

def handle_packet(pkt):
    print(pkt.summary())

print("Dinleniyor... (CTRL+C ile durdur)")
sniff(count=10, prn=handle_packet)