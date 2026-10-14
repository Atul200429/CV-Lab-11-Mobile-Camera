"""
Task 1: Live Motion & Security Alarm Detector
==============================================
Tracks frame differences in real time. If motion above a specific pixel
threshold is detected (e.g., someone walking past the phone camera),
triggers a visual on-screen alert and saves an emergency snapshot to disk.

Key Skills: cv2.absdiff(), frame subtraction, contour area thresholding.

Controls:
    q / ESC  →  Quit
    +/-      →  Increase / decrease sensitivity threshold

Usage:
    python motion_detector.py
    python motion_detector.py --source 0                 # local webcam
    python motion_detector.py --source http://IP:PORT/video
"""

import cv2
import numpy as np
import os
import sys
import argparse
import datetime
import winsound

# ── Add project root to path ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.camera_utils import get_camera, release_and_close, put_text

# ── Configuration ──
MOTION_THRESHOLD = 5000        # minimum contour area to trigger alarm
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "..", "snapshots")
COOLDOWN_SECONDS = 3           # minimum gap between snapshots
BLUR_KERNEL = (21, 21)         # Gaussian blur kernel for noise reduction
DILATE_ITERATIONS = 2          # dilation iterations for filling gaps


def parse_args():
    parser = argparse.ArgumentParser(description="Motion & Security Alarm Detector")
    parser.add_argument("--source", default=None,
                        help="Camera source: int for webcam index, URL for IP stream")
    parser.add_argument("--threshold", type=int, default=MOTION_THRESHOLD,
                        help=f"Contour area threshold (default: {MOTION_THRESHOLD})")
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse source
    source = args.source
    if source is not None:
        try:
            source = int(source)
        except ValueError:
            pass  # keep as string URL

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    cap = get_camera(source=source)
    threshold = args.threshold
    last_snapshot_time = datetime.datetime.min

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read initial frame.")
        release_and_close(cap)
        sys.exit(1)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, BLUR_KERNEL, 0)

    print("=" * 60)
    print("  MOTION & SECURITY ALARM DETECTOR")
    print("=" * 60)
    print(f"  Threshold    : {threshold} px²")
    print(f"  Snapshots    : {os.path.abspath(SNAPSHOT_DIR)}")
    print(f"  Cooldown     : {COOLDOWN_SECONDS}s")
    print("  Controls     : +/- to adjust threshold | q/ESC to quit")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed. Retrying...")
            continue

        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)

        # Compute absolute difference
        frame_diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

        # Dilate to fill gaps
        thresh = cv2.dilate(thresh, None, iterations=DILATE_ITERATIONS)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        max_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < threshold:
                continue

            motion_detected = True
            max_area = max(max_area, area)

            # Draw bounding box around motion
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Status bar
        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

        if motion_detected:
            # ── ALARM STATE ──
            # Red border around entire frame
            cv2.rectangle(frame, (0, 0),
                          (frame.shape[1] - 1, frame.shape[0] - 1),
                          (0, 0, 255), 4)

            put_text(frame, "!! MOTION DETECTED !!", (10, 35),
                     color=(0, 0, 255), scale=1.0, thickness=3)
            put_text(frame, f"Area: {max_area} px^2", (10, 70),
                     color=(0, 0, 255), scale=0.7)
            put_text(frame, f"Threshold: {threshold}", (10, 100),
                     color=(0, 255, 255), scale=0.6)

            # Save snapshot (with cooldown)
            elapsed = (now - last_snapshot_time).total_seconds()
            if elapsed >= COOLDOWN_SECONDS:
                filename = now.strftime("snapshot_%Y%m%d_%H%M%S.jpg")
                filepath = os.path.join(SNAPSHOT_DIR, filename)
                cv2.imwrite(filepath, frame)
                print(f"[ALERT] {timestamp_str} | Motion detected! Snapshot saved: {filepath}")
                last_snapshot_time = now

                # Audible alarm (Windows beep)
                try:
                    winsound.Beep(1000, 300)
                except Exception:
                    pass
        else:
            # ── NORMAL STATE ──
            # Green border
            cv2.rectangle(frame, (0, 0),
                          (frame.shape[1] - 1, frame.shape[0] - 1),
                          (0, 255, 0), 2)
            put_text(frame, "Monitoring...", (10, 35), color=(0, 255, 0), scale=0.8)
            put_text(frame, f"Threshold: {threshold}", (10, 65),
                     color=(0, 255, 255), scale=0.6)

        put_text(frame, timestamp_str, (10, frame.shape[0] - 15),
                 color=(255, 255, 255), scale=0.5)

        # Show frames
        cv2.imshow("Motion Detector - Live Feed", frame)
        cv2.imshow("Frame Difference", thresh)

        # Update previous frame
        prev_gray = gray.copy()

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # q or ESC
            break
        elif key == ord('+') or key == ord('='):
            threshold += 500
            print(f"[INFO] Threshold increased to {threshold}")
        elif key == ord('-') or key == ord('_'):
            threshold = max(500, threshold - 500)
            print(f"[INFO] Threshold decreased to {threshold}")

    print("\n[INFO] Motion detector stopped.")
    release_and_close(cap)


if __name__ == "__main__":
    main()
