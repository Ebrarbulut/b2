import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scapy.all import sniff, IP, TCP, UDP, ICMP  # type: ignore

from scripts.models_unsupervised.standard_autoencoder import StandardAutoencoder


# === GENEL AYARLAR ===
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Flow parametreleri
FLOW_TIMEOUT_SECONDS = 60  # Bir flow bu kadar süredir yeni paket almıyorsa kapat
MAX_OPEN_FLOWS = 10_000    # Çok büyümesini engelle
BATCH_SIZE = 100           # Aynı anda modele gönderilecek flow sayısı


def _load_feature_columns() -> list[str]:
    """
    Eğitimde kullanılan feature kolonlarını advanced_features.csv'den oku.
    """
    features_file = DATA_DIR / "features" / "advanced_features.csv"
    if not features_file.exists():
        raise FileNotFoundError(
            f"advanced_features.csv bulunamadı: {features_file}\n"
            "Önce offline pipeline'ı çalıştırmalısın:\n"
            "  python scripts/merge_and_label_logs.py\n"
            "  python scripts/data_pipeline/advanced_features.py"
        )

    df = pd.read_csv(features_file, nrows=1)
    return list(df.columns)


def _load_standard_autoencoder() -> StandardAutoencoder:
    """
    Daha önce eğitilip kaydedilmiş Standard Autoencoder modelini yükle.

    Yapamadığımız kısım: Bu script içinden modeli tekrar EĞİTME.
    Senin, eğitim sonrasında aşağıdaki dosyaları üretmiş olman gerekiyor:
      - models/standard_autoencoder.keras
      - models/standard_ae_scaler.pkl
      - standard_ae_config.pkl
    Bunlar yoksa load_model() hata verir; o durumda eğitim script'inde
    StandardAutoencoder.save_model(...) çağırıp bu dosyaları üretmen gerekiyor.
    """
    model_path = MODELS_DIR / "standard_autoencoder.keras"
    scaler_path = MODELS_DIR / "standard_ae_scaler.pkl"
    config_path = BASE_DIR / "standard_ae_config.pkl"

    if not model_path.exists() or not scaler_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            "Standard Autoencoder model dosyaları bulunamadı.\n"
            "Beklenen dosyalar:\n"
            f"  - {model_path}\n"
            f"  - {scaler_path}\n"
            f"  - {config_path}\n\n"
            "Lütfen eğitim/optimizasyon adımından sonra şu fonksiyonu çağır:\n"
            "  StandardAutoencoder.save_model(\n"
            "      model_path='models/standard_autoencoder.keras',\n"
            "      scaler_path='models/standard_ae_scaler.pkl',\n"
            "  )\n"
            "ve oluşan standard_ae_config.pkl dosyasını proje kökünde tuttuğundan emin ol."
        )

    return StandardAutoencoder.load_model(
        model_path=model_path,
        scaler_path=scaler_path,
        config_path=config_path,
    )


class RealtimeNIDS:
    """
    Scapy ile gerçek zamanlı trafik yakalayıp
    StandardAutoencoder ile anomali tespiti yapan basit NIDS.
    """

    def __init__(
        self,
        interface: str | None = None,
        flow_timeout: int = FLOW_TIMEOUT_SECONDS,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self.interface = interface
        self.flow_timeout = flow_timeout
        self.batch_size = batch_size

        self.feature_columns = _load_feature_columns()
        self.detector = _load_standard_autoencoder()

        # key: (src_ip, src_port, dst_ip, dst_port, proto)
        self.flows: dict[tuple, dict] = {}
        self.last_flush_time = time.time()

        print("✅ Realtime NIDS hazır.")
        print(f"   Interface: {self.interface or 'varsayılan'}")
        print(f"   Flow timeout: {self.flow_timeout} sn")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Feature sayısı: {len(self.feature_columns)}")

    # === FLOW YÖNETİMİ ===

    @staticmethod
    def _flow_key(pkt):
        if IP not in pkt:
            return None

        ip = pkt[IP]
        proto = "other"
        sport = 0
        dport = 0

        if TCP in pkt:
            proto = "tcp"
            sport = pkt[TCP].sport
            dport = pkt[TCP].dport
        elif UDP in pkt:
            proto = "udp"
            sport = pkt[UDP].sport
            dport = pkt[UDP].dport
        elif ICMP in pkt:
            proto = "icmp"

        return (ip.src, sport, ip.dst, dport, proto)

    def _update_flow(self, pkt) -> None:
        key = self._flow_key(pkt)
        if key is None:
            return

        now = float(pkt.time)
        length = len(pkt)

        if key not in self.flows:
            src_ip, src_p, dst_ip, dst_p, proto = key
            self.flows[key] = {
                "ts_start": now,
                "ts_end": now,
                "id.orig_h": src_ip,
                "id.orig_p": src_p,
                "id.resp_h": dst_ip,
                "id.resp_p": dst_p,
                "proto": proto,
                "orig_bytes": length,
                "resp_bytes": 0,
                "orig_pkts": 1,
                "resp_pkts": 0,
            }
        else:
            f = self.flows[key]
            f["ts_end"] = now
            f["orig_bytes"] += length
            f["orig_pkts"] += 1

        # Güvenlik: çok fazla flow birikirse en eski flow'ları at
        if len(self.flows) > MAX_OPEN_FLOWS:
            # basit çözüm: en eski 10% flow'u sil
            sorted_keys = sorted(
                self.flows.items(), key=lambda kv: kv[1]["ts_end"]
            )
            cutoff = int(len(sorted_keys) * 0.1) or 1
            for k, _ in sorted_keys[:cutoff]:
                del self.flows[k]

    def _build_feature_row(self, flow: dict) -> dict:
        duration = max(flow["ts_end"] - flow["ts_start"], 0.0)
        orig_bytes = float(flow["orig_bytes"])
        resp_bytes = float(flow["resp_bytes"])
        orig_pkts = float(flow["orig_pkts"])
        resp_pkts = float(flow["resp_pkts"])

        total_bytes = orig_bytes + resp_bytes
        total_pkts = orig_pkts + resp_pkts

        bytes_ratio = orig_bytes / resp_bytes if resp_bytes > 0 else 0.0
        pkts_ratio = orig_pkts / resp_pkts if resp_pkts > 0 else 0.0

        # Port entropy ve connections_per_min gibi temporal feature'ları
        # gerçek zamanlıda ilk sürümde 0 geçiyoruz. İstersen bunları
        # daha sonra stateful olarak ekleyebilirsin.
        base_row = {
            "duration": duration,
            "orig_bytes": orig_bytes,
            "resp_bytes": resp_bytes,
            "orig_pkts": orig_pkts,
            "resp_pkts": resp_pkts,
            "total_bytes": total_bytes,
            "total_pkts": total_pkts,
            "bytes_ratio": bytes_ratio,
            "pkts_ratio": pkts_ratio,
            "src_port_entropy": 0.0,
            "dst_port_entropy": 0.0,
            "connections_per_min": 0.0,
        }

        # Sadece modelin beklediği kolonları, doğru sırada ver
        ordered = {col: base_row.get(col, 0.0) for col in self.feature_columns}
        return ordered

    def _flush_flows(self, force: bool = False) -> None:
        """
        Zaman aşımına uğramış veya batch dolduğu için kapanan flow'ları
        modele gönder ve anomalileri yazdır.
        """
        now = time.time()

        to_flush: list[tuple[tuple, dict]] = []
        for key, f in list(self.flows.items()):
            if force or (now - f["ts_end"] > self.flow_timeout):
                to_flush.append((key, f))

        if not to_flush:
            return

        rows: list[dict] = []
        meta: list[dict] = []
        for key, f in to_flush:
            rows.append(self._build_feature_row(f))
            meta.append(f)
            del self.flows[key]

        df = pd.DataFrame(rows)
        X = df.fillna(0.0).values.astype(np.float32)

        # Threshold'u canlı trafikte biraz daha az agresif yapmak için
        # kaydedilen threshold'u bir çarpanla yükseltiyoruz.
        # İstersen bu çarpanı (ör. 1.5, 2.0) terminalde gözlemlerine göre değiştirebilirsin.
        adjusted_threshold = (
            self.detector.threshold * 1.5 if self.detector.threshold is not None else None
        )

        preds, scores = self.detector.predict(X, threshold=adjusted_threshold)

        for f, pred, score in zip(meta, preds, scores):
            if int(pred) == 1:
                print(
                    f"🚨 ANOMALY | {f['id.orig_h']}:{f['id.orig_p']} -> "
                    f"{f['id.resp_h']}:{f['id.resp_p']} | proto={f['proto']} "
                    f"| score={float(score):.6f}"
                )

        self.last_flush_time = now

    # === SCAPY CALLBACK / ANA DÖNGÜ ===

    def _on_packet(self, pkt) -> None:
        self._update_flow(pkt)

        # Timeout'a göre veya çok flow biriktiyse flush et
        if len(self.flows) >= self.batch_size or (
            time.time() - self.last_flush_time > self.flow_timeout
        ):
            self._flush_flows()

    def run(self) -> None:
        print("🔍 Gerçek zamanlı NIDS başlıyor... (CTRL+C ile durdur)")
        print("Bu komutu yönetici olarak çalıştırdığından emin ol.")

        try:
            sniff(
                iface=self.interface,
                prn=self._on_packet,
                store=False,
            )
        except PermissionError:
            print(
                "❌ PermissionError: Paket yakalamak için yeterli yetki yok.\n"
                "PowerShell veya terminali 'Yönetici olarak çalıştır' ile açıp tekrar dene."
            )


def main() -> None:
    # Şimdilik interface'i None bırakıyoruz (Scapy varsayılanı kullanır).
    # İstersen buraya 'Wi-Fi 2' gibi bir interface adı verebilirsin.
    nids = RealtimeNIDS(interface=None)
    try:
        nids.run()
    finally:
        # Program kapanırken elde kalan flow'ları da işle
        nids._flush_flows(force=True)


if __name__ == "__main__":
    main()

