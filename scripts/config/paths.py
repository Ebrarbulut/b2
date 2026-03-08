from pathlib import Path


def get_base_dir() -> Path:
    """
    Proje kök dizinini döndürür.

    Varsayım: Bu dosya `scripts/` altında bir alt pakette bulunuyor ve
    proje kök dizini de bu klasörün bir üstündedir.
    """
    return Path(__file__).resolve().parents[2]


BASE_DIR: Path = get_base_dir()
DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PCAP_DIR: Path = DATA_DIR / "pcap"
LABELED_DIR: Path = DATA_DIR / "labeled"
FEATURE_DIR: Path = DATA_DIR / "features"
MODELS_DIR: Path = BASE_DIR / "models"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"

