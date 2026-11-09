import sys
import numpy as np
import cv2

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from app.detector import ThaiCoinDetector


def generate_synthetic_coins():
    h, w = 600, 800
    img = np.full((h, w, 3), 220, dtype=np.uint8)
    noise = np.random.normal(0, 3, (h, w, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 1. Coin 10 Baht (Radius ~52 px) at (200, 300)
    cv2.circle(img, (200, 300), 52, (190, 195, 200), -1)
    cv2.circle(img, (200, 300), 52, (120, 120, 120), 2)
    cv2.circle(img, (200, 300), 30, (50, 175, 220), -1)
    cv2.putText(img, "10", (185, 308), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 140, 180), 2)

    # 2. Coin 5 Baht (Radius ~48 px) at (400, 300)
    cv2.circle(img, (400, 300), 48, (195, 200, 205), -1)
    cv2.circle(img, (400, 300), 48, (130, 130, 130), 2)
    cv2.putText(img, "5", (390, 308), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (130, 130, 130), 2)

    # 3. Coin 1 Baht (Radius ~40 px) at (600, 300)
    cv2.circle(img, (600, 300), 40, (200, 205, 210), -1)
    cv2.circle(img, (600, 300), 40, (140, 140, 140), 2)
    cv2.putText(img, "1", (592, 308), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 140, 140), 2)

    return img


def test():
    detector = ThaiCoinDetector()

    # --- Test 1: Empty Scene & Non-Coin Rejection ---
    print("[1/2] Testing empty scenes and non-coin rejection...")
    blank = np.full((600, 800, 3), 240, dtype=np.uint8)
    res_blank = detector.detect_and_classify(blank)
    assert res_blank["summary"]["total_coins"] == 0, "Failed: Blank image should detect 0 coins"

    clutter = np.full((600, 800, 3), 210, dtype=np.uint8)
    cv2.circle(clutter, (250, 300), 45, (20, 20, 220), -1)  # Red plastic cap
    cv2.circle(clutter, (500, 300), 45, (220, 40, 20), -1)  # Blue lid
    res_clutter = detector.detect_and_classify(clutter)
    assert res_clutter["summary"]["total_coins"] == 0, "Failed: Non-metallic objects should be rejected"
    print("      [✓] Rejection of empty scenes & non-coins passed!")

    # --- Test 2: Real Coin Detection & Classification ---
    print("[2/2] Testing Thai coins detection & classification...")
    test_img = generate_synthetic_coins()
    result = detector.detect_and_classify(test_img, sensitivity=30)

    for c in result["coins"]:
        print(f"      - Detected: {c['label']} at ({c['x']}, {c['y']}) radius={c['radius']}px, conf={c['confidence']}")

    summary = result["summary"]
    print(f"      Count 10฿: {summary['count_10']}, 5฿: {summary['count_5']}, 1฿: {summary['count_1']}")
    print(f"      Total: {summary['total_coins']} coins, {summary['total_baht']} Baht")

    assert summary["count_10"] == 1, f"Expected 1 10-baht coin, got {summary['count_10']}"
    assert summary["count_5"] == 1, f"Expected 1 5-baht coin, got {summary['count_5']}"
    assert summary["count_1"] == 1, f"Expected 1 1-baht coin, got {summary['count_1']}"
    assert summary["total_baht"] == 16, f"Expected 16 Baht, got {summary['total_baht']}"

    print("[✓] ALL TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test()
