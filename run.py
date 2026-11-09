import sys
import argparse
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import uvicorn

from app.ssl_helper import ensure_ssl_certs, get_local_ip


def main():
    parser = argparse.ArgumentParser(description="Thai Coin Counter & Detector Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen (default: 8000)")
    parser.add_argument("--http", action="store_true", help="Run in plain HTTP mode (No SSL)")
    args = parser.parse_args()

    local_ip = get_local_ip()

    ssl_keyfile = None
    ssl_certfile = None
    protocol = "http"

    if not args.http:
        try:
            cert_path, key_path = ensure_ssl_certs(Path("certs"))
            ssl_keyfile = str(key_path)
            ssl_certfile = str(cert_path)
            protocol = "https"
        except Exception as e:
            print(f"[!] Warning: Could not create SSL certificates ({e}). Falling back to HTTP.")
            protocol = "http"

    print("=" * 65)
    print("      🪙  ระบบตรวจจับและนับเหรียญไทย (THAI COIN COUNTER)  🪙")
    print("=" * 65)
    print(f"[*] สถานะเซิร์ฟเวอร์: กำลังเริ่มต้น...")
    print(f"[*] เปิดใช้งานบนเครื่องคอมพิวเตอร์:")
    print(f"    👉  {protocol}://localhost:{args.port}")
    print(f"    👉  {protocol}://127.0.0.1:{args.port}")
    print()
    print(f"[*] เปิดใช้งานบนโทรศัพท์มือถือ (ต้องต่อ Wi-Fi เดียวกัน):")
    print(f"    👉  {protocol}://{local_ip}:{args.port}")
    print()
    if protocol == "https":
        print("💡 หมายเหตุสำหรับเปิดบนมือถือ:")
        print("   เนื่องจากเป็นใบรับรองความปลอดภัยเฉพาะในวง LAN (Self-signed SSL):")
        print("   - Chrome มือถือ: กด 'ขั้นสูง' (Advanced) -> กด 'ไปยังไซต์นี้' (Proceed)")
        print("   - Safari มือถือ: กด 'แสดงรายละเอียด' -> กด 'เข้าชมเว็บไซต์นี้'")
        print("   - จากนั้นกดยินยอมให้เบราว์เซอร์ใช้งานกล้องได้ทันทีครับ")
    print("=" * 65)
    print("กด Ctrl+C เพื่อหยุดการทำงาน\n")

    uvicorn_kwargs = {
        "app": "app.main:app",
        "host": args.host,
        "port": args.port,
        "reload": False,
        "log_level": "info",
    }

    if ssl_keyfile and ssl_certfile:
        uvicorn_kwargs["ssl_keyfile"] = ssl_keyfile
        uvicorn_kwargs["ssl_certfile"] = ssl_certfile

    uvicorn.run(**uvicorn_kwargs)


if __name__ == "__main__":
    main()
