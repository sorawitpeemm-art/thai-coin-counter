import sys
import base64
import cv2
import numpy as np

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from fastapi.testclient import TestClient
from app.main import app
from test_detector import generate_synthetic_coins

client = TestClient(app)


def test_routes():
    res = client.get("/")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    print("[✓] GET / -> 200 OK")

    res = client.get("/api/info")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    info = res.json()
    print(f"[✓] GET /api/info -> {info}")
    assert info.get("yolo_available") is True, "Expected yolo_available to be True"

    res = client.get("/api/config")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    config = res.json()
    print(f"[✓] GET /api/config -> {config}")

    # Test updating config to yolo
    res = client.post("/api/config", json={"engine": "yolo"})
    assert res.status_code == 200
    assert res.json()["config"]["engine"] == "yolo"
    print("[✓] POST /api/config -> switched to 'yolo'")

    # Test /api/detect with YOLO on synthetic coins
    img = generate_synthetic_coins()
    _, buffer = cv2.imencode('.jpg', img)
    b64_str = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

    res = client.post("/api/detect", data={"image_base64": b64_str, "engine": "yolo"})
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    print(f"[✓] POST /api/detect (YOLO) -> {data['summary']}")
    assert data.get("engine") == "yolo"
    assert data["summary"]["total_coins"] > 0, "Expected at least 1 coin detected by YOLO"

    # Test switching back to local
    res = client.post("/api/config", json={"engine": "local"})
    assert res.status_code == 200
    assert res.json()["config"]["engine"] == "local"
    print("[✓] POST /api/config -> switched back to 'local'")

    # Test /api/detect with local OpenCV
    res = client.post("/api/detect", data={"image_base64": b64_str, "engine": "local"})
    assert res.status_code == 200
    data_local = res.json()
    print(f"[✓] POST /api/detect (Local OpenCV) -> {data_local['summary']}")
    assert data_local.get("engine") == "local"

    print("\n🎉 [✓] All server route & multi-engine detection tests passed successfully!")


if __name__ == "__main__":
    test_routes()
