"""
Gerçek zamanlı paket tabanlı ajan (prototip).

Amaç:
- Windows makinede (Npcap kurulu iken) ağ arayüzünden canlı paket toplamak.
- Kısa aralıklarla (ör. 5 sn) basit akış/istatistik feature'ları üretmek.
- Bu feature vektörünü backend'deki /api/score endpoint'ine POST etmek.
- Gelen anomali skoruna göre terminalde "normal / şüpheli / anomalik" durumu göstermek.

Önemli:
- Windows'ta Npcap kurulu olmalı (https://nmap.org/npcap/).
- Script yönetici olarak çalıştırılmalı (Run as Administrator).
- Bu, ürünleştirme için ilk prototiptir; feature set'i demo amaçlıdır.
"""

import argparse
import time
from typing import List

import numpy as np
import requests
from scapy.all import sniff  # type: ignore[import]
from scapy.layers.inet import IP, TCP, UDP  # type: ignore[import]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Canlı paket ajanı (Npcap + scapy).")
  parser.add_argument(
      "--backend-url",
      type=str,
      default="http://127.0.0.1:8000",
      help="FastAPI backend base URL (varsayılan: http://127.0.0.1:8000).",
  )
  parser.add_argument(
      "--iface",
      type=str,
      default=None,
      help="Dinlenecek arayüz adı (boş bırakılırsa varsayılan arayüz).",
  )
  parser.add_argument(
      "--window-seconds",
      type=int,
      default=5,
      help="Her analiz penceresinin süresi (saniye).",
  )
  return parser.parse_args()


def collect_window_packets(iface: str | None, window_seconds: int):
  """Belirtilen arayüzden window_seconds süreyle paketleri topla."""
  packets = sniff(timeout=window_seconds, iface=iface)
  return packets


def build_feature_vector(packets) -> List[float]:
  """
  Toplanan paketlerden basit bir 12 boyutlu feature vektörü çıkar.

  Not: Bu özellikler, mevcut modelin advanced_features'ıyla birebir aynı olmayabilir;
  bu prototipte amaç, canlı veriyi sayısal bir vektöre döküp /api/score akışını
  ürünleştirmektir. Modeli bu feature'lara göre yeniden eğitmek ayrı bir adımdır.
  """
  total_packets = len(packets)
  if total_packets == 0:
    return [0.0] * 12

  sizes: List[int] = []
  tcp_count = 0
  udp_count = 0
  other_count = 0
  src_ips: set[str] = set()
  dst_ips: set[str] = set()
  tcp_syn_count = 0

  for pkt in packets:
    if IP in pkt:
      ip_layer = pkt[IP]
      src_ips.add(ip_layer.src)
      dst_ips.add(ip_layer.dst)
      sizes.append(len(pkt))
    else:
      sizes.append(len(pkt))

    if TCP in pkt:
      tcp_count += 1
      flags = pkt[TCP].flags
      # SYN (0x02) bayraklı paket sayısı
      if flags & 0x02:
        tcp_syn_count += 1
    elif UDP in pkt:
      udp_count += 1
    else:
      other_count += 1

  sizes_arr = np.array(sizes, dtype=float)
  total_bytes = float(np.sum(sizes_arr))

  tcp_ratio = tcp_count / total_packets
  udp_ratio = udp_count / total_packets
  other_ratio = other_count / total_packets

  unique_src = len(src_ips)
  unique_dst = len(dst_ips)

  mean_size = float(np.mean(sizes_arr))
  std_size = float(np.std(sizes_arr))
  max_size = float(np.max(sizes_arr))
  min_size = float(np.min(sizes_arr))

  syn_ratio = tcp_syn_count / total_packets

  features: List[float] = [
      float(total_packets),
      float(total_bytes),
      tcp_ratio,
      udp_ratio,
      other_ratio,
      float(unique_src),
      float(unique_dst),
      mean_size,
      std_size,
      max_size,
      min_size,
      syn_ratio,
  ]
  return features


def fetch_score(backend_url: str, features: List[float]) -> float:
  payload = {"features": [features]}
  resp = requests.post(f"{backend_url.rstrip('/')}/api/score", json=payload, timeout=10)
  resp.raise_for_status()
  data = resp.json()
  scores = data.get("scores", [])
  return float(scores[0]) if scores else 0.0


def describe_score(score: float) -> str:
  if score < 0.3:
    return "normal"
  if score < 0.7:
    return "şüpheli"
  return "anomalik"


def main() -> None:
  args = parse_args()
  backend_url = args.backend_url
  iface = args.iface
  window_seconds = args.window_seconds

  print(f"[packet-agent] Backend: {backend_url}")
  print(f"[packet-agent] Interface: {iface or 'varsayılan'}")
  print(f"[packet-agent] Pencere süresi: {window_seconds} sn")
  print("[packet-agent] Başlamak için Ctrl+C ile durdurulabilir sonsuz döngüye giriliyor.")

  while True:
    try:
      packets = collect_window_packets(iface, window_seconds)
      features = build_feature_vector(packets)
      score = fetch_score(backend_url, features)
      level = describe_score(score)
      print(
          f"[packet-agent] {window_seconds} sn içinde {int(features[0])} paket,"
          f" skor={score:.3f} → durum: {level}"
      )
    except KeyboardInterrupt:
      print("\n[packet-agent] Kullanıcı tarafından durduruldu.")
      break
    except Exception as e:  # noqa: BLE001
      print(f"[packet-agent] Hata: {e}")
      time.sleep(window_seconds)


if __name__ == "__main__":
  main()

