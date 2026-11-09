import base64
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.detector import ThaiCoinDetector
from app.yolo_detector import LocalYoloCoinDetector
from app.roboflow_detector import RoboflowCoinDetector
from app.config_manager import load_config, save_config
from app.ssl_helper import get_local_ip

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CoinDetectorApp")

app = FastAPI(title="Thai Coin Counter & Detector", version="2.1.0")

# Enable CORS for local and mobile browser connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration and detectors
current_config = load_config()
local_detector = ThaiCoinDetector()
yolo_detector = LocalYoloCoinDetector()
roboflow_detector = RoboflowCoinDetector(api_key=current_config.get("roboflow_api_key"))

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_file = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.get("/api/info")
async def get_info():
    """Return system, network, and active engine info."""
    local_ip = get_local_ip()
    return {
        "status": "online",
        "local_ip": local_ip,
        "engine": current_config.get("engine", "local"),
        "yolo_available": yolo_detector.is_available(),
        "has_roboflow_key": bool(current_config.get("roboflow_api_key")),
        "roboflow_model": f"{RoboflowCoinDetector.MODEL_ID}/{RoboflowCoinDetector.VERSION}",
        "supported_coins": ["10 Baht", "5 Baht", "2 Baht", "1 Baht", "Satang"],
    }


@app.get("/api/config")
async def get_config():
    """Get current engine configuration without exposing raw key."""
    key = current_config.get("roboflow_api_key", "")
    masked_key = f"{key[:4]}...{key[-4:]}" if len(key) >= 8 else ("***" if key else "")
    return {
        "engine": current_config.get("engine", "local"),
        "yolo_available": yolo_detector.is_available(),
        "has_roboflow_key": bool(key),
        "masked_key": masked_key,
        "roboflow_model": "sut-jpi3i/thai-coin-detection",
        "confidence_threshold": current_config.get("confidence_threshold", 40),
    }


@app.post("/api/config")
async def update_config(request: Request):
    """Update runtime configuration (switch engine, update Roboflow API key)."""
    global current_config, roboflow_detector
    try:
        body = await request.json()
        if "engine" in body:
            current_config["engine"] = body["engine"]
        if "roboflow_api_key" in body and body["roboflow_api_key"] is not None:
            new_key = body["roboflow_api_key"].strip()
            if new_key:
                current_config["roboflow_api_key"] = new_key
                roboflow_detector.api_key = new_key
        if "confidence_threshold" in body:
            current_config["confidence_threshold"] = int(body["confidence_threshold"])

        save_config(current_config)
        logger.info(
            f"Configuration updated: engine={current_config.get('engine')}, "
            f"has_key={bool(current_config.get('roboflow_api_key'))}"
        )
        return {"status": "ok", "config": await get_config()}
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)


def decode_image_base64(data_url: str) -> Optional[np.ndarray]:
    """Decodes a base64 Data URL to an OpenCV BGR image."""
    try:
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        image_bytes = base64.b64decode(data_url)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        logger.error(f"Decode image failed: {e}")
        return None


@app.post("/api/detect")
async def detect_snapshot(
    file: UploadFile = File(None),
    image_base64: str = Form(None),
    engine: str = Form(None),
    sensitivity: int = Form(32),
    min_radius: int = Form(18),
    max_radius: int = Form(180),
    guide_scale: float = Form(None),
    guide_center_x: int = Form(None),
    guide_center_y: int = Form(None),
    guide_radius: int = Form(None),
):
    """
    HTTP POST endpoint for high-resolution single-shot capture.
    Accepts multipart file or base64 string.
    Supports engines: 'local', 'yolo', 'roboflow'.
    """
    try:
        if file is not None:
            contents = await file.read()
            image_base64 = "data:image/jpeg;base64," + base64.b64encode(contents).decode("utf-8")
            np_arr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        elif image_base64:
            img = decode_image_base64(image_base64)
        else:
            return JSONResponse({"error": "No image provided"}, status_code=400)

        if img is None:
            return JSONResponse({"error": "Could not decode image"}, status_code=400)

        selected_engine = engine or current_config.get("engine", "local")
        guide_center = (guide_center_x, guide_center_y) if (guide_center_x is not None and guide_center_y is not None) else None

        # 1. Roboflow Cloud Engine
        if selected_engine == "roboflow" and roboflow_detector.api_key:
            try:
                conf = current_config.get("confidence_threshold", 40)
                result = await roboflow_detector.detect(image_base64, confidence=conf)
                return JSONResponse(result)
            except Exception as e:
                logger.warning(f"Roboflow detection failed: {e}. Falling back to local.")

        # 2. Local YOLO Offline AI Engine
        if selected_engine == "yolo" and yolo_detector.is_available():
            try:
                conf = current_config.get("confidence_threshold", 40) / 100.0
                result = yolo_detector.detect_and_classify(
                    img,
                    confidence=conf,
                    guide_center=guide_center,
                    guide_radius=guide_radius,
                )
                return JSONResponse(result)
            except Exception as e:
                logger.warning(f"Local YOLO detection failed: {e}. Falling back to OpenCV.")

        # 3. Local OpenCV Engine (Default)
        result = local_detector.detect_and_classify(
            img,
            min_radius=min_radius,
            max_radius=max_radius,
            sensitivity=sensitivity,
            guide_scale=guide_scale,
            guide_center=guide_center,
            guide_radius=guide_radius,
        )
        result["engine"] = "local"
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Error processing snapshot: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.websocket("/ws")
async def websocket_detect(websocket: WebSocket):
    """
    Real-time WebSocket endpoint for low-latency live camera streaming.
    Supports dynamic engine switching between 'local', 'yolo', and 'roboflow'.
    """
    await websocket.accept()
    logger.info("WebSocket client connected.")

    try:
        while True:
            text = await websocket.receive_text()
            data = json.loads(text)

            image_data = data.get("image")
            if not image_data:
                continue

            requested_engine = data.get("engine") or current_config.get("engine", "local")

            # Route to Roboflow if requested and API key is present
            if requested_engine == "roboflow" and roboflow_detector.api_key:
                try:
                    conf = int(data.get("confidence_threshold", current_config.get("confidence_threshold", 40)))
                    result = await roboflow_detector.detect(image_data, confidence=conf)
                    await websocket.send_text(json.dumps(result))
                    continue
                except Exception as e:
                    logger.warning(f"Roboflow WS error: {e}. Falling back.")

            # Route to Local YOLO Offline AI
            if requested_engine == "yolo" and yolo_detector.is_available():
                try:
                    conf = float(data.get("confidence_threshold", current_config.get("confidence_threshold", 40))) / 100.0
                    guide_center = data.get("guide_center")
                    guide_radius = data.get("guide_radius")
                    if guide_center is not None and guide_radius is not None:
                        guide_center = (int(guide_center[0]), int(guide_center[1]))
                        guide_radius = int(guide_radius)

                    img = decode_image_base64(image_data)
                    if img is not None:
                        result = yolo_detector.detect_and_classify(
                            img,
                            confidence=conf,
                            guide_center=guide_center,
                            guide_radius=guide_radius,
                        )
                        await websocket.send_text(json.dumps(result))
                        continue
                except Exception as e:
                    logger.warning(f"YOLO WS error: {e}. Falling back to OpenCV.")

            # Route to Local OpenCV Processing
            sensitivity = int(data.get("sensitivity", 30))
            min_radius = int(data.get("min_radius", 18))
            max_radius = int(data.get("max_radius", 180))
            guide_scale = data.get("guide_scale")
            if guide_scale is not None:
                guide_scale = float(guide_scale)

            guide_center = data.get("guide_center")
            guide_radius = data.get("guide_radius")
            if guide_center is not None and guide_radius is not None:
                guide_center = (int(guide_center[0]), int(guide_center[1]))
                guide_radius = int(guide_radius)

            # Decode frame
            img = decode_image_base64(image_data)
            if img is None:
                await websocket.send_text(json.dumps({"error": "Decode failed"}))
                continue

            # Process frame with local coin detector
            result = local_detector.detect_and_classify(
                img,
                min_radius=min_radius,
                max_radius=max_radius,
                sensitivity=sensitivity,
                guide_scale=guide_scale,
                guide_center=guide_center,
                guide_radius=guide_radius,
            )
            result["engine"] = "local"

            # Send back JSON result
            await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
