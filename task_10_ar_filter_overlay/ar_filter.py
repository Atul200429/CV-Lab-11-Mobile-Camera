"""
Task 10: Real-Time Augmented Reality (AR) Filter Overlay
=========================================================
Detects facial features using Mediapipe Face Mesh, then renders
custom 2D visual overlays (virtual glasses, hats, mustache, masks)
mapped to facial landmark anchors in real time.

Key Skills: Landmark localization, image overlaying with alpha channels
            (transparency blending), affine transformations.

Controls:
    q / ESC  →  Quit
    1-5      →  Switch AR filter
    0        →  Remove all filters
    s        →  Save snapshot with filter

Filters:
    1 → Sunglasses
    2 → Devil Horns
    3 → Mustache
    4 → Clown Nose
    5 → Full Face Mask (all combined)
    0 → No filter

Usage:
    python ar_filter.py
    python ar_filter.py --source 0
"""

import cv2
import numpy as np
import os
import sys
import argparse
import datetime
import math

# ── Add project root to path ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.camera_utils import get_camera, release_and_close, put_text

try:
    import mediapipe as mp
except ImportError:
    print("[ERROR] mediapipe is required. Install: pip install mediapipe")
    sys.exit(1)

# ── Mediapipe setup ──
mp_face_mesh = mp.solutions.face_mesh

# ── Key facial landmark indices (Mediapipe 468 landmarks) ──
# Forehead center
FOREHEAD = 10
# Nose tip
NOSE_TIP = 1
# Nose bridge (between eyes)
NOSE_BRIDGE = 6
# Left eye outer, inner
LEFT_EYE_OUTER = 263
LEFT_EYE_INNER = 362
# Right eye outer, inner
RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133
# Mouth corners
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
# Chin
CHIN = 152
# Upper lip center
UPPER_LIP = 0
# Left/Right temple
LEFT_TEMPLE = 454
RIGHT_TEMPLE = 234


def parse_args():
    parser = argparse.ArgumentParser(description="AR Filter Overlay")
    parser.add_argument("--source", default=None)
    return parser.parse_args()


def get_landmark_point(landmarks, index, frame_w, frame_h):
    """Get pixel coordinates for a landmark index."""
    lm = landmarks[index]
    return int(lm.x * frame_w), int(lm.y * frame_h)


def draw_sunglasses(frame, landmarks, frame_w, frame_h):
    """Draw cool sunglasses over the eyes."""
    # Get eye positions
    left_outer = get_landmark_point(landmarks, LEFT_EYE_OUTER, frame_w, frame_h)
    right_outer = get_landmark_point(landmarks, RIGHT_EYE_OUTER, frame_w, frame_h)
    left_inner = get_landmark_point(landmarks, LEFT_EYE_INNER, frame_w, frame_h)
    right_inner = get_landmark_point(landmarks, RIGHT_EYE_INNER, frame_w, frame_h)
    nose_bridge = get_landmark_point(landmarks, NOSE_BRIDGE, frame_w, frame_h)

    # Calculate dimensions
    eye_width = int(np.linalg.norm(
        np.array(left_outer) - np.array(right_outer)
    ))
    glass_h = int(eye_width * 0.35)
    half_w = eye_width // 2 + 15

    # Calculate angle
    dx = left_outer[0] - right_outer[0]
    dy = left_outer[1] - right_outer[1]
    angle = math.degrees(math.atan2(dy, dx))

    center_x = (left_outer[0] + right_outer[0]) // 2
    center_y = (left_outer[1] + right_outer[1]) // 2

    # Draw frame of glasses
    # Left lens
    left_center = ((center_x + left_outer[0]) // 2,
                    (center_y + left_outer[1]) // 2)
    lens_w = int(eye_width * 0.35)
    lens_h = int(glass_h * 0.8)

    # Create overlay with transparency
    overlay = frame.copy()

    # Left lens (dark tinted)
    cv2.ellipse(overlay, left_center, (lens_w, lens_h), angle,
                0, 360, (40, 20, 20), -1)
    cv2.ellipse(overlay, left_center, (lens_w, lens_h), angle,
                0, 360, (0, 0, 0), 2)

    # Right lens
    right_center = ((center_x + right_outer[0]) // 2,
                     (center_y + right_outer[1]) // 2)
    cv2.ellipse(overlay, right_center, (lens_w, lens_h), angle,
                0, 360, (40, 20, 20), -1)
    cv2.ellipse(overlay, right_center, (lens_w, lens_h), angle,
                0, 360, (0, 0, 0), 2)

    # Bridge
    cv2.line(overlay,
             (left_center[0] + lens_w, left_center[1]),
             (right_center[0] - lens_w, right_center[1]),
             (0, 0, 0), 3)

    # Temple arms
    cv2.line(overlay,
             (left_center[0] - lens_w, left_center[1]),
             (left_outer[0] - 10, left_outer[1]),
             (0, 0, 0), 3)
    cv2.line(overlay,
             (right_center[0] + lens_w, right_center[1]),
             (right_outer[0] + 10, right_outer[1]),
             (0, 0, 0), 3)

    # Blend
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)


def draw_devil_horns(frame, landmarks, frame_w, frame_h):
    """Draw devil horns on the forehead."""
    forehead = get_landmark_point(landmarks, FOREHEAD, frame_w, frame_h)
    left_temple = get_landmark_point(landmarks, LEFT_TEMPLE, frame_w, frame_h)
    right_temple = get_landmark_point(landmarks, RIGHT_TEMPLE, frame_w, frame_h)

    face_width = abs(left_temple[0] - right_temple[0])
    horn_height = int(face_width * 0.5)

    # Left horn
    left_base = (forehead[0] + face_width // 4, forehead[1] - 10)
    left_tip = (left_base[0] + 20, left_base[1] - horn_height)
    left_inner = (left_base[0] - 15, left_base[1])

    pts_left = np.array([left_base, left_tip, left_inner], dtype=np.int32)
    cv2.fillPoly(frame, [pts_left], (0, 0, 200))
    cv2.polylines(frame, [pts_left], True, (0, 0, 100), 2)

    # Right horn
    right_base = (forehead[0] - face_width // 4, forehead[1] - 10)
    right_tip = (right_base[0] - 20, right_base[1] - horn_height)
    right_inner = (right_base[0] + 15, right_base[1])

    pts_right = np.array([right_base, right_tip, right_inner], dtype=np.int32)
    cv2.fillPoly(frame, [pts_right], (0, 0, 200))
    cv2.polylines(frame, [pts_right], True, (0, 0, 100), 2)


def draw_mustache(frame, landmarks, frame_w, frame_h):
    """Draw a curly mustache under the nose."""
    nose_tip = get_landmark_point(landmarks, NOSE_TIP, frame_w, frame_h)
    upper_lip = get_landmark_point(landmarks, UPPER_LIP, frame_w, frame_h)
    mouth_left = get_landmark_point(landmarks, MOUTH_LEFT, frame_w, frame_h)
    mouth_right = get_landmark_point(landmarks, MOUTH_RIGHT, frame_w, frame_h)

    # Mustache center
    center_x = (nose_tip[0] + upper_lip[0]) // 2
    center_y = (nose_tip[1] + upper_lip[1]) // 2

    mouth_width = abs(mouth_left[0] - mouth_right[0])
    half_w = int(mouth_width * 0.6)
    h = int(mouth_width * 0.15)

    # Draw using bezier-like curves
    # Left side
    pts_left = []
    for t in np.linspace(0, 1, 20):
        x = int(center_x - t * half_w)
        y = int(center_y + h * math.sin(t * math.pi * 1.5))
        pts_left.append((x, y))

    # Right side
    pts_right = []
    for t in np.linspace(0, 1, 20):
        x = int(center_x + t * half_w)
        y = int(center_y + h * math.sin(t * math.pi * 1.5))
        pts_right.append((x, y))

    # Draw thick mustache
    for i in range(1, len(pts_left)):
        cv2.line(frame, pts_left[i-1], pts_left[i], (30, 30, 30), 4)
    for i in range(1, len(pts_right)):
        cv2.line(frame, pts_right[i-1], pts_right[i], (30, 30, 30), 4)

    # Curl ends
    if pts_left:
        cv2.circle(frame, pts_left[-1], 5, (30, 30, 30), -1)
    if pts_right:
        cv2.circle(frame, pts_right[-1], 5, (30, 30, 30), -1)


def draw_clown_nose(frame, landmarks, frame_w, frame_h):
    """Draw a red clown nose."""
    nose_tip = get_landmark_point(landmarks, NOSE_TIP, frame_w, frame_h)
    nose_bridge = get_landmark_point(landmarks, NOSE_BRIDGE, frame_w, frame_h)

    # Nose size based on face proportions
    nose_size = int(abs(nose_tip[1] - nose_bridge[1]) * 0.6)
    nose_size = max(15, min(40, nose_size))

    # Draw with gradient effect
    for i in range(nose_size, 0, -1):
        intensity = int(255 * (i / nose_size))
        cv2.circle(frame, nose_tip, i,
                   (0, 0, min(255, intensity + 50)), -1)

    # Highlight
    highlight_pos = (nose_tip[0] - nose_size // 3,
                     nose_tip[1] - nose_size // 3)
    cv2.circle(frame, highlight_pos, nose_size // 4,
               (180, 180, 255), -1)


# ── Filter registry ──
FILTERS = {
    0: {"name": "No Filter", "funcs": []},
    1: {"name": "Sunglasses", "funcs": [draw_sunglasses]},
    2: {"name": "Devil Horns", "funcs": [draw_devil_horns]},
    3: {"name": "Mustache", "funcs": [draw_mustache]},
    4: {"name": "Clown Nose", "funcs": [draw_clown_nose]},
    5: {"name": "Full Combo", "funcs": [draw_sunglasses, draw_devil_horns,
                                         draw_mustache, draw_clown_nose]},
}


def main():
    args = parse_args()

    source = args.source
    if source is not None:
        try:
            source = int(source)
        except ValueError:
            pass

    cap = get_camera(source=source)
    current_filter = 1
    save_dir = os.path.join(os.path.dirname(__file__), "..", "snapshots")
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("  AR FILTER OVERLAY")
    print("=" * 60)
    print("  Filters:")
    for k, v in FILTERS.items():
        print(f"    {k} → {v['name']}")
    print("  Controls: 0-5 switch filter | s save | q/ESC quit")
    print("=" * 60)

    with mp_face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            faces_detected = 0

            if results.multi_face_landmarks:
                faces_detected = len(results.multi_face_landmarks)

                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark

                    # Apply selected filter(s)
                    for filter_func in FILTERS[current_filter]["funcs"]:
                        try:
                            filter_func(frame, landmarks, frame_w, frame_h)
                        except Exception:
                            pass  # Gracefully skip if landmarks are off-screen

            # Header
            cv2.rectangle(frame, (0, 0), (frame_w, 40), (30, 30, 30), -1)
            put_text(frame, f"AR Filter: [{current_filter}] "
                          f"{FILTERS[current_filter]['name']} | "
                          f"Faces: {faces_detected}",
                     (10, 28), color=(0, 255, 0), scale=0.6)

            cv2.imshow("AR Filter Overlay", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key in [ord(str(i)) for i in range(6)]:
                current_filter = int(chr(key))
                print(f"[INFO] Filter: {FILTERS[current_filter]['name']}")
            elif key == ord('s'):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ar_filter_{timestamp}.png"
                cv2.imwrite(os.path.join(save_dir, filename), frame)
                print(f"  ✔ Snapshot saved: {filename}")

    print("\n[INFO] AR filter overlay stopped.")
    release_and_close(cap)


if __name__ == "__main__":
    main()
