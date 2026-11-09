import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

logger = logging.getLogger("LocalYoloDetector")

# Class mapping from the trained Thai coins model:
# names: {0: '5', 1: '1', 2: '10', 3: '2', 4: '.50'}
YOLO_CLASSES = {
    0: {"name": "5",   "label": "5฿",   "value": 5.0,  "color": "#0EA5E9"},
    1: {"name": "1",   "label": "1฿",   "value": 1.0,  "color": "#10B981"},
    2: {"name": "10",  "label": "10฿",  "value": 10.0, "color": "#F59E0B"},
    3: {"name": "2",   "label": "2฿",   "value": 2.0,  "color": "#EC4899"},
    4: {"name": ".50", "label": "50สต.", "value": 0.5,  "color": "#8B5CF6"},
}


class LocalYoloCoinDetector:
    """
    Local YOLO Deep Learning Detector for Thai Coins.
    Runs completely offline on the user's PC with models/thai_coins_yolo.pt.
    No API key or internet connection required!
    """

    def __init__(self, weights_path: str = "models/thai_coins_yolo.pt"):
        self.weights_path = Path(weights_path)
        self.model = None
        self._load_model()

    def _load_model(self):
        if not self.weights_path.exists():
            logger.warning(f"YOLO weights not found at {self.weights_path}")
            return
        try:
            base_dir = Path(__file__).resolve().parent.parent
            yolo_lib_path = base_dir / "yolov5_lib"
            if yolo_lib_path.exists() and str(yolo_lib_path) not in sys.path:
                sys.path.insert(0, str(yolo_lib_path))
            else:
                hub_path = Path.home() / ".cache" / "torch" / "hub" / "ultralytics_yolov5_master"
                if hub_path.exists() and str(hub_path) not in sys.path:
                    sys.path.insert(0, str(hub_path))

            from models.experimental import attempt_load
            from models.common import AutoShape

            raw_model = attempt_load(str(self.weights_path), device="cpu", fuse=False)
            self.model = AutoShape(raw_model)
            logger.info("Local Thai Coin YOLO model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
            self.model = None

    def is_available(self) -> bool:
        return self.model is not None

    def detect_and_classify(
        self,
        image_bgr: np.ndarray,
        confidence: float = 0.40,
        guide_center: Optional[tuple] = None,
        guide_radius: Optional[int] = None,
    ) -> Dict[str, Any]:
        h, w = image_bgr.shape[:2]
        if self.model is None or h == 0 or w == 0:
            return {
                "coins": [],
                "summary": self._empty_summary(),
                "frame_width": w,
                "frame_height": h,
                "engine": "yolo",
            }

        # OpenCV BGR -> RGB for AutoShape
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.model.conf = float(confidence)
        results = self.model(img_rgb, size=640)

        coins = []
        count_10 = 0
        count_5 = 0
        count_2 = 0
        count_1 = 0
        count_satang = 0
        total_baht = 0.0

        df = results.pandas().xyxy[0]

        for _, row in df.iterrows():
            x1 = float(row["xmin"])
            y1 = float(row["ymin"])
            x2 = float(row["xmax"])
            y2 = float(row["ymax"])
            conf = float(row["confidence"])
            cls_id = int(row["class"])
            cls_name = str(row["name"])

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            aspect = bw / bh
            if aspect < 0.72 or aspect > 1.38:
                continue  # Coins are strictly circular (reject elongated non-coin objects)
            radius = max(bw, bh) / 2.0
            if radius < 12 or radius > min(w, h) * 0.48:
                continue

            # Guide Reticle ROI filter
            if guide_center is not None and guide_radius is not None:
                gx, gy = guide_center
                dist = np.hypot(cx - gx, cy - gy)
                if dist > guide_radius * 1.05:
                    continue

            info = YOLO_CLASSES.get(cls_id)
            if not info:
                for c_info in YOLO_CLASSES.values():
                    if c_info["name"] == cls_name:
                        info = c_info
                        break
            if not info:
                info = {"name": cls_name, "label": f"{cls_name}฿", "value": 0.0, "color": "#94A3B8"}

            val = info["value"]
            total_baht += val

            if info["name"] == "10":
                count_10 += 1
                type_val = 10
            elif info["name"] == "5":
                count_5 += 1
                type_val = 5
            elif info["name"] == "2":
                count_2 += 1
                type_val = 2
            elif info["name"] == "1":
                count_1 += 1
                type_val = 1
            elif info["name"] == ".50":
                count_satang += 1
                type_val = 0.5
            else:
                type_val = 0

            coins.append({
                "id": len(coins) + 1,
                "type": type_val,
                "label": info["label"],
                "class_name": info["name"],
                "x": int(round(cx)),
                "y": int(round(cy)),
                "radius": int(round(radius)),
                "color": info["color"],
                "confidence": round(conf, 2),
            })

        summary = {
            "count_10": count_10,
            "count_5": count_5,
            "count_2": count_2,
            "count_1": count_1,
            "count_satang": count_satang,
            "total_coins": len(coins),
            "total_baht": round(total_baht, 2),
        }

        return {
            "coins": coins,
            "summary": summary,
            "frame_width": w,
            "frame_height": h,
            "engine": "yolo",
        }

    def _empty_summary(self) -> Dict[str, Any]:
        return {
            "count_10": 0,
            "count_5": 0,
            "count_2": 0,
            "count_1": 0,
            "count_satang": 0,
            "total_coins": 0,
            "total_baht": 0,
        }
