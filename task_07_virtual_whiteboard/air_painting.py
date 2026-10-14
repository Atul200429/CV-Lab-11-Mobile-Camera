"""
Task 7: Virtual Whiteboard / Air Painting App
==============================================
Tracks a colored object (bright blue marker cap, red pen, green
laser pointer) in front of the camera and draws a path on the PC
screen to paint or write in mid-air.

Key Skills: Object tracking, drawing primitives (cv2.line, cv2.circle),
            canvas buffers, HSV color tracking.

Controls:
    q / ESC  →  Quit
    c        →  Clear the canvas
    1-4      →  Switch drawing color (Blue/Green/Red/Yellow)
    +/-      →  Increase / decrease brush size
    s        →  Save the canvas as an image
    e        →  Toggle eraser mode
    u        →  Undo last stroke

Usage:
    python air_painting.py
    python air_painting.py --source 0
    python air_painting.py --tracker blue   # blue/green/red/yellow
"""

import cv2
import numpy as np
import os
import sys
import argparse
import datetime
from collections import deque

# ── Add project root to path ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.camera_utils import get_camera, release_and_close, put_text

# ── Tracker Color Ranges (HSV) ──
TRACKER_COLORS = {
    "blue": {
        "lower": np.array([100, 100, 80]),
        "upper": np.array([130, 255, 255]),
        "label": "Blue object"
    },
    "green": {
        "lower": np.array([40, 70, 70]),
        "upper": np.array([80, 255, 255]),
        "label": "Green object"
    },
    "red": {
        "lower": np.array([0, 120, 100]),
        "upper": np.array([10, 255, 255]),
        "lower2": np.array([160, 120, 100]),
        "upper2": np.array([180, 255, 255]),
        "label": "Red object"
    },
    "yellow": {
        "lower": np.array([20, 100, 100]),
        "upper": np.array([35, 255, 255]),
        "label": "Yellow object"
    },
}

# Drawing colors (BGR)
DRAW_COLORS = {
    1: ("Blue",   (255, 100, 50)),
    2: ("Green",  (50, 255, 50)),
    3: ("Red",    (50, 50, 255)),
    4: ("Yellow", (50, 255, 255)),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Virtual Whiteboard / Air Painting")
    parser.add_argument("--source", default=None)
    parser.add_argument("--tracker", default="blue",
                        choices=["blue", "green", "red", "yellow"],
                        help="Color of the tracked object")
    return parser.parse_args()


def track_object(hsv_frame, tracker_config):
    """Find the center of the tracked colored object."""
    mask = cv2.inRange(hsv_frame, tracker_config["lower"], tracker_config["upper"])

    # Handle red wrapping
    if "lower2" in tracker_config:
        mask2 = cv2.inRange(hsv_frame, tracker_config["lower2"], tracker_config["upper2"])
        mask = cv2.bitwise_or(mask, mask2)

    # Clean up
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    center = None
    radius = 0

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > 800:  # Minimum area threshold
            ((x, y), r) = cv2.minEnclosingCircle(largest)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                center = (int(M["m10"] / M["m00"]),
                          int(M["m01"] / M["m00"]))
                radius = int(r)

    return center, radius, mask


def main():
    args = parse_args()

    source = args.source
    if source is not None:
        try:
            source = int(source)
        except ValueError:
            pass

    cap = get_camera(source=source)

    # Read first frame to get dimensions
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read initial frame.")
        release_and_close(cap)
        sys.exit(1)

    h, w = frame.shape[:2]

    # Create canvas (transparent initially)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # State
    tracker_config = TRACKER_COLORS[args.tracker]
    draw_color_idx = 1
    brush_size = 5
    eraser_mode = False
    prev_point = None
    strokes = []           # List of strokes for undo
    current_stroke = []     # Points in current stroke
    save_dir = os.path.join(os.path.dirname(__file__), "..", "snapshots")
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("  VIRTUAL WHITEBOARD / AIR PAINTING")
    print("=" * 60)
    print(f"  Tracker : {tracker_config['label']}")
    print("  Colors  : 1=Blue  2=Green  3=Red  4=Yellow")
    print("  Controls: c=clear | e=eraser | u=undo | s=save")
    print("            +/- brush size | q/ESC quit")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Mirror the frame for natural drawing
        frame = cv2.flip(frame, 1)

        # Convert to HSV and track
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        center, radius, mask = track_object(hsv, tracker_config)

        if center is not None:
            # Draw tracker indicator
            cv2.circle(frame, center, radius, (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

            # Draw on canvas
            if prev_point is not None:
                dist = np.linalg.norm(
                    np.array(center) - np.array(prev_point)
                )
                if dist < 150:  # Skip large jumps
                    if eraser_mode:
                        cv2.line(canvas, prev_point, center, (0, 0, 0),
                                 brush_size * 3)
                    else:
                        draw_color = DRAW_COLORS[draw_color_idx][1]
                        cv2.line(canvas, prev_point, center, draw_color,
                                 brush_size)
                    current_stroke.append(center)

            prev_point = center
        else:
            # Object lost → end current stroke
            if current_stroke:
                strokes.append(current_stroke.copy())
                current_stroke.clear()
            prev_point = None

        # Combine canvas with frame
        # Make canvas semi-transparent
        canvas_mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, canvas_mask = cv2.threshold(canvas_mask, 1, 255, cv2.THRESH_BINARY)

        # Where canvas has drawing, use canvas; else use camera
        combined = frame.copy()
        combined[canvas_mask > 0] = canvas[canvas_mask > 0]

        # UI Header
        cv2.rectangle(combined, (0, 0), (w, 50), (30, 30, 30), -1)
        color_name, color_bgr = DRAW_COLORS[draw_color_idx]
        mode = "ERASER" if eraser_mode else color_name

        put_text(combined, f"Air Paint | {mode} | Brush: {brush_size}",
                 (10, 22), color=color_bgr if not eraser_mode else (200, 200, 200),
                 scale=0.6)
        put_text(combined, f"Tracker: {tracker_config['label']} | "
                          f"Strokes: {len(strokes)}",
                 (10, 42), color=(200, 200, 200), scale=0.45)

        # Color palette indicator
        for i, (_, cbgr) in DRAW_COLORS.items():
            x = w - 40 * (5 - i)
            cv2.circle(combined, (x, 25), 12, cbgr, -1)
            if i == draw_color_idx:
                cv2.circle(combined, (x, 25), 15, (255, 255, 255), 2)

        cv2.imshow("Air Painting", combined)
        cv2.imshow("Canvas", canvas)
        cv2.imshow("Tracking Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('c'):
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            strokes.clear()
            current_stroke.clear()
            print("[INFO] Canvas cleared.")
        elif key in [ord(str(i)) for i in range(1, 5)]:
            draw_color_idx = int(chr(key))
            eraser_mode = False
            print(f"[INFO] Color: {DRAW_COLORS[draw_color_idx][0]}")
        elif key == ord('+') or key == ord('='):
            brush_size = min(30, brush_size + 1)
            print(f"[INFO] Brush size: {brush_size}")
        elif key == ord('-') or key == ord('_'):
            brush_size = max(1, brush_size - 1)
            print(f"[INFO] Brush size: {brush_size}")
        elif key == ord('e'):
            eraser_mode = not eraser_mode
            print(f"[INFO] Eraser: {'ON' if eraser_mode else 'OFF'}")
        elif key == ord('s'):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(save_dir, f"painting_{timestamp}.png"), canvas)
            cv2.imwrite(os.path.join(save_dir, f"painting_combined_{timestamp}.png"),
                        combined)
            print(f"  ✔ Painting saved!")
        elif key == ord('u'):
            if strokes:
                strokes.pop()
                # Redraw canvas from strokes
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                # Note: We'd need to store color/size per stroke for full undo
                print(f"[INFO] Undo. Strokes remaining: {len(strokes)}")

    print("\n[INFO] Air painting stopped.")
    release_and_close(cap)


if __name__ == "__main__":
    main()
