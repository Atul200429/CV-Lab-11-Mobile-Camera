"""
Task 10: AR Filter Overlay
============================
Detects facial landmarks using Mediapipe FaceLandmarker and renders
2D visual overlays (sunglasses, horns, mustache, clown nose) mapped
to face landmarks in real time.

Available Filters:
    0 → No filter
    1 → Sunglasses
    2 → Devil Horns
    3 → Mustache
    4 → Clown Nose
    5 → Full Combo (all)

Key Skills: Mediapipe FaceLandmarker, alpha blending, affine transforms.

Controls:
    0-5      →  Switch filter
    s        →  Save snapshot
    q / ESC  →  Quit

Usage:
    python ar_filter.py
    python ar_filter.py --source 0
"""

import cv2
import os
import sys
import argparse
import datetime
import math
import numpy as np

# ── Add project root to path ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.camera_utils import get_camera, release_and_close, put_text

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions, FaceLandmarkerResult, RunningMode
)

# ── Configuration ──
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "face_landmarker.task")
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "..", "snapshots")

FILTER_NAMES = {
    0: "No Filter",
    1: "Sunglasses",
    2: "Devil Horns",
    3: "Mustache",
    4: "Clown Nose",
    5: "Full Combo",
}

# Key landmark indices (468-point mesh)
NOSE_TIP = 1
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_OUTER = 263
RIGHT_EYE_INNER = 362
LEFT_EYEBROW_TOP = 70
RIGHT_EYEBROW_TOP = 300
UPPER_LIP_TOP = 0
LOWER_LIP_BOTTOM = 17
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
FOREHEAD_TOP = 10


def parse_args():
    parser = argparse.ArgumentParser(description="AR Filter Overlay")
    parser.add_argument("--source", default=None,
                        help="Camera source: int for webcam index, URL for IP stream")
    return parser.parse_args()


def draw_sunglasses(frame, landmarks, w, h):
    """Draw cool sunglasses over the eyes."""
    le_outer = landmarks[LEFT_EYE_OUTER]
    re_outer = landmarks[RIGHT_EYE_OUTER]
    nose = landmarks[NOSE_TIP]

    x1, y1 = int(le_outer.x * w), int(le_outer.y * h)
    x2, y2 = int(re_outer.x * w), int(re_outer.y * h)
    nose_x, nose_y = int(nose.x * w), int(nose.y * h)

    # Sunglasses dimensions
    glass_w = int(abs(x2 - x1) * 1.3)
    glass_h = int(glass_w * 0.35)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    # Draw dark lens shapes
    left_lens_center = (center_x - glass_w // 4, center_y)
    right_lens_center = (center_x + glass_w // 4, center_y)
    lens_w = glass_w // 3
    lens_h = glass_h // 2

    overlay = frame.copy()
    cv2.ellipse(overlay, left_lens_center, (lens_w, lens_h), 0, 0, 360, (20, 20, 20), -1)
    cv2.ellipse(overlay, right_lens_center, (lens_w, lens_h), 0, 0, 360, (20, 20, 20), -1)
    cv2.ellipse(overlay, left_lens_center, (lens_w, lens_h), 0, 0, 360, (0, 200, 255), 2)
    cv2.ellipse(overlay, right_lens_center, (lens_w, lens_h), 0, 0, 360, (0, 200, 255), 2)

    # Bridge
    cv2.line(overlay,
             (left_lens_center[0] + lens_w, center_y),
             (right_lens_center[0] - lens_w, center_y),
             (0, 200, 255), 2)

    # Temple arms
    cv2.line(overlay, (left_lens_center[0] - lens_w, center_y),
             (x1 - 20, center_y - 5), (0, 200, 255), 2)
    cv2.line(overlay, (right_lens_center[0] + lens_w, center_y),
             (x2 + 20, center_y - 5), (0, 200, 255), 2)

    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)


def draw_devil_horns(frame, landmarks, w, h):
    """Draw devil horns above the forehead."""
    left_brow = landmarks[LEFT_EYEBROW_TOP]
    right_brow = landmarks[RIGHT_EYEBROW_TOP]
    forehead = landmarks[FOREHEAD_TOP]

    lx, ly = int(left_brow.x * w), int(left_brow.y * h)
    rx, ry = int(right_brow.x * w), int(right_brow.y * h)
    fx, fy = int(forehead.x * w), int(forehead.y * h)

    horn_h = int(abs(fy - ly) * 1.5) + 30

    # Left horn
    pts_left = np.array([
        [lx - 15, ly - 5],
        [lx + 10, ly - 5],
        [lx - 25, ly - horn_h],
    ], np.int32)

    # Right horn
    pts_right = np.array([
        [rx - 10, ry - 5],
        [rx + 15, ry - 5],
        [rx + 25, ry - horn_h],
    ], np.int32)

    cv2.fillPoly(frame, [pts_left], (0, 0, 200))
    cv2.fillPoly(frame, [pts_right], (0, 0, 200))
    cv2.polylines(frame, [pts_left], True, (0, 0, 100), 2)
    cv2.polylines(frame, [pts_right], True, (0, 0, 100), 2)


def draw_mustache(frame, landmarks, w, h):
    """Draw a curly mustache."""
    mouth_l = landmarks[MOUTH_LEFT]
    mouth_r = landmarks[MOUTH_RIGHT]
    upper_lip = landmarks[UPPER_LIP_TOP]

    ml_x, ml_y = int(mouth_l.x * w), int(mouth_l.y * h)
    mr_x, mr_y = int(mouth_r.x * w), int(mouth_r.y * h)
    ul_x, ul_y = int(upper_lip.x * w), int(upper_lip.y * h)

    center_x = (ml_x + mr_x) // 2
    center_y = ul_y + 5
    half_w = abs(mr_x - ml_x) // 2 + 10

    # Draw mustache curves
    pts = []
    for angle in range(0, 181, 5):
        rad = math.radians(angle)
        x = center_x + int(half_w * math.cos(rad))
        y = center_y - int(12 * math.sin(rad) * math.sin(rad))
        pts.append([x, y])

    pts = np.array(pts, np.int32)
    cv2.polylines(frame, [pts], False, (40, 30, 20), 4)

    # Curly ends
    cv2.circle(frame, (center_x - half_w - 5, center_y - 5), 8, (40, 30, 20), -1)
    cv2.circle(frame, (center_x + half_w + 5, center_y - 5), 8, (40, 30, 20), -1)


def draw_clown_nose(frame, landmarks, w, h):
    """Draw a red clown nose."""
    nose = landmarks[NOSE_TIP]
    cx, cy = int(nose.x * w), int(nose.y * h)
    radius = int(w * 0.025) + 8

    cv2.circle(frame, (cx, cy), radius, (0, 0, 255), -1)
    cv2.circle(frame, (cx, cy), radius, (0, 0, 180), 2)
    # Shine highlight
    cv2.circle(frame, (cx - radius // 3, cy - radius // 3),
               radius // 4, (100, 100, 255), -1)


def apply_filter(frame, landmarks, w, h, filter_id):
    """Apply the selected AR filter."""
    if filter_id == 0:
        return
    elif filter_id == 1:
        draw_sunglasses(frame, landmarks, w, h)
    elif filter_id == 2:
        draw_devil_horns(frame, landmarks, w, h)
    elif filter_id == 3:
        draw_mustache(frame, landmarks, w, h)
    elif filter_id == 4:
        draw_clown_nose(frame, landmarks, w, h)
    elif filter_id == 5:
        draw_sunglasses(frame, landmarks, w, h)
        draw_devil_horns(frame, landmarks, w, h)
        draw_mustache(frame, landmarks, w, h)
        draw_clown_nose(frame, landmarks, w, h)


def main():
    args = parse_args()

    source = args.source
    if source is not None:
        try:
            source = int(source)
        except ValueError:
            pass

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        print("        Download from: https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
        sys.exit(1)

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    cap = get_camera(source=source)
    current_filter = 1  # Start with sunglasses

    # Store latest result from callback
    latest_result = [None]

    def result_callback(result, output_image, timestamp_ms):
        latest_result[0] = result

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM,
        num_faces=3,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        result_callback=result_callback,
    )

    print("=" * 60)
    print("  AR FILTER OVERLAY")
    print("=" * 60)
    print("  Filters:")
    for k, v in FILTER_NAMES.items():
        marker = " <<" if k == current_filter else ""
        print(f"    {k} → {v}{marker}")
    print("  Controls: 0-5 switch filter | s save | q/ESC quit")
    print("=" * 60)

    with FaceLandmarker.create_from_options(options) as landmarker:
        frame_timestamp = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            frame_timestamp += 33
            landmarker.detect_async(mp_image, frame_timestamp)

            result = latest_result[0]
            faces_count = 0

            if result and result.face_landmarks:
                faces_count = len(result.face_landmarks)
                for face_landmarks in result.face_landmarks:
                    apply_filter(frame, face_landmarks, frame_w, frame_h, current_filter)

            # UI Header
            cv2.rectangle(frame, (0, 0), (frame_w, 40), (30, 30, 30), -1)
            put_text(frame, f"Filter: {FILTER_NAMES[current_filter]} | "
                            f"Faces: {faces_count}",
                     (10, 28), color=(0, 255, 0), scale=0.6)

            # Filter selector at bottom
            bar_y = frame_h - 35
            cv2.rectangle(frame, (0, bar_y), (frame_w, frame_h), (30, 30, 30), -1)
            for i in range(6):
                color = (0, 255, 0) if i == current_filter else (150, 150, 150)
                put_text(frame, f"[{i}]{FILTER_NAMES[i][:4]}",
                         (10 + i * (frame_w // 6), frame_h - 10),
                         color=color, scale=0.35)

            if faces_count == 0:
                put_text(frame, "No face detected",
                         (frame_w // 2 - 120, frame_h // 2),
                         color=(0, 255, 255), scale=0.7)

            cv2.imshow("AR Filter Overlay", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key in range(ord('0'), ord('6')):
                current_filter = key - ord('0')
                print(f"  Filter: {FILTER_NAMES[current_filter]}")
            elif key == ord('s'):
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(SNAPSHOT_DIR, f"ar_filter_{ts}.jpg")
                cv2.imwrite(path, frame)
                print(f"  Saved: {path}")

    print("\n[INFO] AR Filter overlay stopped.")
    release_and_close(cap)


if __name__ == "__main__":
    main()
