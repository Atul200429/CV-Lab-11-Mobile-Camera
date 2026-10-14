"""
Task 2: Mobile QR Code & Barcode Scanner for Inventory
=======================================================
Turns the smartphone into a portable handheld scanner. Reads QR/barcodes
in real time from the stream, decodes the data, and logs the decoded
items into a CSV file with timestamps.

Key Skills: OpenCV QRCodeDetector, pyzbar, CSV file manipulation.

Controls:
    q / ESC  →  Quit
    c        →  Clear the on-screen log
    s        →  Show CSV log path

Usage:
    python qr_barcode_scanner.py
    python qr_barcode_scanner.py --source 0
"""

import cv2
import csv
import os
import sys
import argparse
import datetime
from collections import OrderedDict

# ── Add project root to path ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.camera_utils import get_camera, release_and_close, put_text

# Try importing pyzbar (more robust); fall back to OpenCV QRCodeDetector
USE_PYZBAR = False
try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    from pyzbar.pyzbar import ZBarSymbol
    USE_PYZBAR = True
    print("[INFO] Using pyzbar for decoding.")
except ImportError:
    print("[INFO] pyzbar not found. Using OpenCV QRCodeDetector (QR codes only).")

# ── Configuration ──
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "inventory_log")
CSV_FILENAME = "inventory_log.csv"
DUPLICATE_COOLDOWN = 5  # seconds before same code is logged again


def parse_args():
    parser = argparse.ArgumentParser(description="QR Code & Barcode Scanner")
    parser.add_argument("--source", default=None,
                        help="Camera source: int for webcam index, URL for IP stream")
    return parser.parse_args()


def init_csv(filepath):
    """Create CSV with headers if it doesn't exist."""
    if not os.path.exists(filepath):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Type", "Data"])
    return filepath


def log_to_csv(filepath, code_type, data):
    """Append a decoded entry to the CSV log."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, code_type, data])
    return timestamp


def decode_pyzbar(frame):
    """Decode using pyzbar – supports QR codes, barcodes (EAN, Code128, etc.)."""
    results = []
    decoded_objects = pyzbar_decode(frame)
    for obj in decoded_objects:
        data = obj.data.decode("utf-8", errors="replace")
        code_type = obj.type
        # Draw polygon around detected code
        points = obj.polygon
        if len(points) == 4:
            pts = [(p.x, p.y) for p in points]
            for i in range(4):
                cv2.line(frame, pts[i], pts[(i + 1) % 4], (0, 255, 0), 3)
        else:
            rect = obj.rect
            cv2.rectangle(frame, (rect.left, rect.top),
                          (rect.left + rect.width, rect.top + rect.height),
                          (0, 255, 0), 3)
        results.append((code_type, data))
    return results


def decode_opencv(frame, detector):
    """Decode using OpenCV QRCodeDetector (QR codes only)."""
    results = []
    data, points, _ = detector.detectAndDecode(frame)
    if data and points is not None:
        points = points[0].astype(int)
        for i in range(4):
            cv2.line(frame, tuple(points[i]), tuple(points[(i + 1) % 4]),
                     (0, 255, 0), 3)
        results.append(("QRCODE", data))
    return results


def main():
    args = parse_args()

    source = args.source
    if source is not None:
        try:
            source = int(source)
        except ValueError:
            pass

    os.makedirs(LOG_DIR, exist_ok=True)
    csv_path = init_csv(os.path.join(LOG_DIR, CSV_FILENAME))

    cap = get_camera(source=source)
    qr_detector = None if USE_PYZBAR else cv2.QRCodeDetector()

    # Track recently scanned codes to avoid duplicates
    recent_scans = OrderedDict()  # data -> last_scan_time
    scan_history = []             # for on-screen display (last 10)
    total_scanned = 0

    print("=" * 60)
    print("  QR CODE & BARCODE SCANNER")
    print("=" * 60)
    print(f"  Decoder  : {'pyzbar (multi-format)' if USE_PYZBAR else 'OpenCV (QR only)'}")
    print(f"  CSV Log  : {os.path.abspath(csv_path)}")
    print(f"  Cooldown : {DUPLICATE_COOLDOWN}s between duplicate scans")
    print("  Controls : q/ESC to quit | c to clear log display")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        now = datetime.datetime.now()

        # Decode
        if USE_PYZBAR:
            results = decode_pyzbar(frame)
        else:
            results = decode_opencv(frame, qr_detector)

        for code_type, data in results:
            # Check cooldown
            if data in recent_scans:
                elapsed = (now - recent_scans[data]).total_seconds()
                if elapsed < DUPLICATE_COOLDOWN:
                    continue

            # Log the scan
            recent_scans[data] = now
            timestamp = log_to_csv(csv_path, code_type, data)
            total_scanned += 1

            entry = f"[{timestamp}] {code_type}: {data}"
            scan_history.append(entry)
            if len(scan_history) > 10:
                scan_history.pop(0)

            print(f"  ✔ Scanned: {code_type} → {data}")

        # ── Draw UI overlay ──
        # Header
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (40, 40, 40), -1)
        put_text(frame, f"Scanner | Total: {total_scanned}", (10, 28),
                 color=(0, 255, 0), scale=0.7)

        # Recent scans panel (right side)
        if scan_history:
            panel_h = min(len(scan_history) * 25 + 10, frame.shape[0] - 50)
            panel_y_start = 50
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, panel_y_start),
                          (frame.shape[1] - 5, panel_y_start + panel_h),
                          (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            for i, entry in enumerate(scan_history[-8:]):
                y = panel_y_start + 20 + i * 25
                put_text(frame, entry[:70], (10, y),
                         color=(255, 255, 255), scale=0.4, thickness=1)

        # Crosshair in center
        cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
        cv2.line(frame, (cx - 30, cy), (cx + 30, cy), (0, 255, 255), 1)
        cv2.line(frame, (cx, cy - 30), (cx, cy + 30), (0, 255, 255), 1)

        cv2.imshow("QR/Barcode Scanner", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('c'):
            scan_history.clear()
            print("[INFO] On-screen log cleared.")
        elif key == ord('s'):
            print(f"[INFO] CSV log at: {os.path.abspath(csv_path)}")

    print(f"\n[INFO] Scanner stopped. Total scanned: {total_scanned}")
    print(f"[INFO] CSV log saved to: {os.path.abspath(csv_path)}")
    release_and_close(cap)


if __name__ == "__main__":
    main()
