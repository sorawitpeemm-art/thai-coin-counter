import json
from pathlib import Path
from typing import Dict, Any

CONFIG_FILE = Path("config.json")

DEFAULT_CONFIG: Dict[str, Any] = {
    "engine": "yolo",  # "yolo" (Offline Thai Coin AI) or "opencv" or "roboflow"
    "roboflow_api_key": "",
    "roboflow_model": "thai-coin-detection",
    "roboflow_version": "1",
    "confidence_threshold": 40,
}


def load_config() -> Dict[str, Any]:
    if not CONFIG_FILE.exists():
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        cfg = DEFAULT_CONFIG.copy()
        cfg.update(data)
        return cfg
    except Exception:
        return DEFAULT_CONFIG.copy()


def save_config(cfg: Dict[str, Any]) -> None:
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
