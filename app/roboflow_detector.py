import base64
import logging
from typing import Dict, Any, List, Optional
import httpx
import cv2
import numpy as np

logger = logging.getLogger("RoboflowDetector")


class RoboflowCoinDetector:
    """
    Client for sut-jpi3i/thai-coin-detection on Roboflow Universe.
    Detects: 10 Baht, 5 Baht, 2 Baht, 1 Baht, .50 Satang, .25 Satang.
    """

    MODEL_ID = "thai-coin-detection"
    VERSION = "1"
    BASE_URL = "https://detect.roboflow.com"

    # Color map for bounding rings
    COLOR_MAP = {
        "10 Baht":    "#F59E0B",  # Amber/Gold
        "5 Baht":     "#0EA5E9",  # Sky Blue
        "2 Baht":     "#EC4899",  # Pink
        "1 Baht":     "#10B981",  # Emerald Green
        ".50 Satang": "#8B5CF6",  # Purple
        ".25 Satang": "#F97316",  # Orange
        "Coin":       "#94A3B8",  # Gray
    }

    # Denomination values in THB
    VALUE_MAP = {
        "10 Baht":    10.0,
        "5 Baht":     5.0,
        "2 Baht":     2.0,
        "1 Baht":     1.0,
        ".50 Satang": 0.50,
        ".25 Satang": 0.25,
        "Coin":       0.0,
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._http_client: Optional[httpx.AsyncClient] = None

    async def get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=10.0)
        return self._http_client

    async def detect(
        self,
        image_base64_data: str,
        confidence: int = 40,
        overlap: int = 30,
    ) -> Dict[str, Any]:
        """
        Send image to Roboflow inference API and parse detections.
        """
        if not self.api_key:
            raise ValueError("Roboflow API Key is not configured.")

        # Clean base64 string
        clean_base64 = image_base64_data
        if "," in clean_base64:
            clean_base64 = clean_base64.split(",", 1)[1]

        url = f"{self.BASE_URL}/{self.MODEL_ID}/{self.VERSION}"
        params = {
            "api_key": self.api_key,
            "confidence": confidence,
            "overlap": overlap,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        client = await self.get_client()
        response = await client.post(url, params=params, data=clean_base64, headers=headers)

        if response.status_code != 200:
            error_msg = response.text
            logger.error(f"Roboflow API error ({response.status_code}): {error_msg}")
            raise RuntimeError(f"Roboflow API error: {error_msg}")

        data = response.json()
        return self._format_predictions(data)

    def _format_predictions(self, roboflow_res: Dict[str, Any]) -> Dict[str, Any]:
        predictions = roboflow_res.get("predictions", [])
        image_meta = roboflow_res.get("image", {})
        frame_w = image_meta.get("width", 640)
        frame_h = image_meta.get("height", 480)

        coins = []
        count_10 = 0
        count_5 = 0
        count_2 = 0
        count_1 = 0
        count_satang = 0
        total_baht = 0.0

        for i, pred in enumerate(predictions):
            cls_name = pred.get("class", "Coin")
            conf = float(pred.get("confidence", 0.0))
            x = float(pred.get("x", 0))
            y = float(pred.get("y", 0))
            w = float(pred.get("width", 0))
            h = float(pred.get("height", 0))

            # Approximate radius as average half-dimension
            radius = int(round((w + h) / 4.0))

            val = self.VALUE_MAP.get(cls_name, 0.0)
            total_baht += val

            # Update specific counters
            if cls_name == "10 Baht":
                count_10 += 1
                type_val = 10
                label = "10฿"
            elif cls_name == "5 Baht":
                count_5 += 1
                type_val = 5
                label = "5฿"
            elif cls_name == "2 Baht":
                count_2 += 1
                type_val = 2
                label = "2฿"
            elif cls_name == "1 Baht":
                count_1 += 1
                type_val = 1
                label = "1฿"
            elif "Satang" in cls_name:
                count_satang += 1
                type_val = 0.5 if "50" in cls_name else 0.25
                label = "50สต." if "50" in cls_name else "25สต."
            else:
                type_val = 0
                label = "เหรียญ"

            coins.append({
                "id": i + 1,
                "type": type_val,
                "label": label,
                "class_name": cls_name,
                "x": int(round(x)),
                "y": int(round(y)),
                "radius": radius,
                "width": int(round(w)),
                "height": int(round(h)),
                "color": self.COLOR_MAP.get(cls_name, "#94A3B8"),
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
            "frame_width": frame_w,
            "frame_height": frame_h,
            "engine": "roboflow",
            "model": f"{self.MODEL_ID}/{self.VERSION}",
        }
