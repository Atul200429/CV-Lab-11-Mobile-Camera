"""
Task 6: Portable Color & Defect Inspector
==========================================
Points the camera at objects and isolates specific color ranges in
HSV color space to detect:
  • Ripe vs. unripe fruit (red/green detection)
  • Defective parts on a mini assembly line
  • Custom color ranges defined by the user

Key Skills: RGB→HSV conversion, color thresholding (cv2.inRange), masking.

Controls:
    q / ESC   →  Quit
    1-6       →  Switch preset color filter
    c         →  Enter custom HSV range mode
    m         →  Toggle mask / overlay view
    s         →  Save current analysis frame

Usage:
    python color_inspector.py
    python color_inspector.py --source 0
"""

import cv2
import numpy as np
import os
import sys
import argparse
import datetime

# ── Add project root to path ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.camera_utils import get_camera, release_and_close, put_text

# ── Color Presets (HSV ranges) ──
COLOR_PRESETS = {
    1: {
        "name": "Red (Ripe Fruit)",
        "lower1": np.array([0, 100, 100]),
        "upper1": np.array([10, 255, 255]),
        "lower2": np.array([160, 100, 100]),  # Red wraps around in HSV
        "upper2": np.array([180, 255, 255]),
        "color_bgr": (0, 0, 255),
    },
    2: {
        "name": "Green (Unripe Fruit)",
        "lower1": np.array([35, 50, 50]),
        "upper1": np.array([85, 255, 255]),
        "lower2": None,
        "upper2": None,
        "color_bgr": (0, 255, 0),
    },
    3: {
        "name": "Blue (Defect Marker)",
        "lower1": np.array([100, 100, 50]),
        "upper1": np.array([130, 255, 255]),
        "lower2": None,
        "upper2": None,
        "color_bgr": (255, 0, 0),
    },
    4: {
        "name": "Yellow (Caution Zone)",
        "lower1": np.array([20, 100, 100]),
        "upper1": np.array([35, 255, 255]),
        "lower2": None,
        "upper2": None,
        "color_bgr": (0, 255, 255),
    },
    5: {
        "name": "Orange",
        "lower1": np.array([10, 100, 100]),
        "upper1": np.array([20, 255, 255]),
        "lower2": None,
        "upper2": None,
        "color_bgr": (0, 165, 255),
    },
    6: {
        "name": "Purple/Violet",
        "lower1": np.array([130, 50, 50]),
        "upper1": np.array([160, 255, 255]),
        "lower2": None,
        "upper2": None,
        "color_bgr": (255, 0, 128),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Color & Defect Inspector")
    parser.add_argument("--source", default=None)
    return parser.parse_args()


def create_color_mask(hsv_frame, preset):
    """Create a binary mask for the specified color range."""
    mask = cv2.inRange(hsv_frame, preset["lower1"], preset["upper1"])

    if preset["lower2"] is not None:
        mask2 = cv2.inRange(hsv_frame, preset["lower2"], preset["upper2"])
        mask = cv2.bitwise_or(mask, mask2)

    # Clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def analyze_mask(frame, mask, color_bgr, label):
    """Find contours in mask and annotate frame with analysis."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    total_area = 0
    object_count = 0
    min_contour_area = 500  # Ignore noise

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue

        object_count += 1
        total_area += area

        # Draw contour and bounding box
        cv2.drawContours(frame, [contour], -1, color_bgr, 2)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)

        # Label each detected region
        cv2.putText(frame, f"{label} #{object_count}",
                    (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color_bgr, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Area: {area}px",
                    (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Frame coverage percentage
    frame_area = frame.shape[0] * frame.shape[1]
    coverage = (total_area / frame_area) * 100 if frame_area > 0 else 0

    return object_count, total_area, coverage


def main():
    args = parse_args()

    source = args.source
    if source is not None:
        try:
            source = int(source)
        except ValueError:
            pass

    cap = get_camera(source=source)
    current_preset = 1
    show_overlay = True  # True=overlay, False=mask only
    save_dir = os.path.join(os.path.dirname(__file__), "..", "snapshots")
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("  PORTABLE COLOR & DEFECT INSPECTOR")
    print("=" * 60)
    print("  Presets:")
    for k, v in COLOR_PRESETS.items():
        print(f"    {k} → {v['name']}")
    print("  Controls: 1-6 presets | m toggle mask | s save | q/ESC quit")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        preset = COLOR_PRESETS[current_preset]

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask
        mask = create_color_mask(hsv, preset)

        # Analyze and annotate
        display = frame.copy()
        count, area, coverage = analyze_mask(
            display, mask, preset["color_bgr"], preset["name"].split()[0]
        )

        # Create colored mask visualization
        colored_mask = cv2.bitwise_and(frame, frame, mask=mask)

        # Header bar
        cv2.rectangle(display, (0, 0), (display.shape[1], 55), (30, 30, 30), -1)
        put_text(display, f"Filter: [{current_preset}] {preset['name']}",
                 (10, 22), color=preset["color_bgr"], scale=0.6)
        put_text(display, f"Objects: {count} | Coverage: {coverage:.1f}% | "
                          f"Area: {area}px",
                 (10, 45), color=(255, 255, 255), scale=0.5)

        # Quality indicator
        if coverage > 50:
            status = "HIGH DENSITY"
            status_color = (0, 255, 0)
        elif coverage > 20:
            status = "MODERATE"
            status_color = (0, 255, 255)
        elif coverage > 5:
            status = "LOW"
            status_color = (0, 165, 255)
        else:
            status = "MINIMAL"
            status_color = (128, 128, 128)

        put_text(display, status, (display.shape[1] - 150, 22),
                 color=status_color, scale=0.6)

        # Show windows
        cv2.imshow("Color Inspector - Live", display)
        cv2.imshow("Color Mask", colored_mask)
        cv2.imshow("Binary Mask", mask)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key in [ord(str(i)) for i in range(1, 7)]:
            current_preset = int(chr(key))
            print(f"[INFO] Switched to: {COLOR_PRESETS[current_preset]['name']}")
        elif key == ord('m'):
            show_overlay = not show_overlay
        elif key == ord('s'):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(save_dir, f"inspect_{timestamp}.png"), display)
            cv2.imwrite(os.path.join(save_dir, f"mask_{timestamp}.png"), colored_mask)
            print(f"  ✔ Saved inspection snapshot.")

    print("\n[INFO] Color inspector stopped.")
    release_and_close(cap)


if __name__ == "__main__":
    main()
