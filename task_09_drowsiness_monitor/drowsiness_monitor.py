"""
Task 9: Drowsiness / Blink Monitor
====================================
Calculates Eye Aspect Ratio (EAR) from facial landmarks.
If EAR stays below threshold for too long, triggers an alarm.

EAR Formula:
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    - EAR ≈ 0.3 when eyes open
    - EAR ≈ 0.0 when eyes closed

Key Skills: Mediapipe FaceLandmarker, EAR calculation, audible alarm.

Controls:
    q / ESC  →  Quit
    +/-      →  Adjust EAR threshold
    r        →  Reset counters

Usage:
    python drowsiness_monitor.py
    python drowsiness_monitor.py --source 0 --ear-threshold 0.20
"""

import cv2
import os
import sys
import argparse
import time
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

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# ── Configuration ──
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "face_landmarker.task")
DEFAULT_EAR_THRESHOLD = 0.22
CONSECUTIVE_FRAMES_ALERT = 20  # frames below threshold to trigger alert
ALERT_BEEP_FREQ = 2500
ALERT_BEEP_DURATION = 200

# Eye landmark indices for MediaPipe FaceLandmarker (468 mesh)
# Left eye: [33, 160, 158, 133, 153, 144]
# Right eye: [362, 385, 387, 263, 373, 380]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def parse_args():
    parser = argparse.ArgumentParser(description="Drowsiness / Blink Monitor")
    parser.add_argument("--source", default=None,
                        help="Camera source: int for webcam index, URL for IP stream")
    parser.add_argument("--ear-threshold", type=float, default=DEFAULT_EAR_THRESHOLD,
                        help=f"EAR threshold for closed eyes (default: {DEFAULT_EAR_THRESHOLD})")
    return parser.parse_args()


def euclidean(p1, p2):
    """Calculate Euclidean distance between two landmarks."""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def calculate_ear(landmarks, eye_indices):
    """Calculate Eye Aspect Ratio for one eye."""
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    vertical_1 = euclidean(p2, p6)
    vertical_2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    if horizontal == 0:
        return 0.0
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def draw_eye(frame, landmarks, eye_indices, w, h, color=(0, 255, 0)):
    """Draw eye contour on frame."""
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    pts = np.array(points, dtype=np.int32)
    cv2.polylines(frame, [pts], True, color, 1)


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

    cap = get_camera(source=source)
    ear_threshold = args.ear_threshold
    frame_counter = 0  # consecutive low-EAR frames
    blink_counter = 0
    alert_active = False
    ear_history = []  # for mini graph

    # Store latest result from callback
    latest_result = [None]

    def result_callback(result, output_image, timestamp_ms):
        latest_result[0] = result

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        result_callback=result_callback,
    )

    print("=" * 60)
    print("  DROWSINESS / BLINK MONITOR")
    print("=" * 60)
    print(f"  EAR Threshold : {ear_threshold:.2f}")
    print(f"  Alert after   : {CONSECUTIVE_FRAMES_ALERT} frames below threshold")
    print("  Controls      : +/- adjust threshold | r reset | q/ESC quit")
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

            ear = 0.0
            face_detected = False

            result = latest_result[0]
            if result and result.face_landmarks:
                face_landmarks = result.face_landmarks[0]
                face_detected = True

                # Calculate EAR for both eyes
                left_ear = calculate_ear(face_landmarks, LEFT_EYE)
                right_ear = calculate_ear(face_landmarks, RIGHT_EYE)
                ear = (left_ear + right_ear) / 2.0

                # Draw eye contours
                eye_color = (0, 255, 0) if ear >= ear_threshold else (0, 0, 255)
                draw_eye(frame, face_landmarks, LEFT_EYE, frame_w, frame_h, eye_color)
                draw_eye(frame, face_landmarks, RIGHT_EYE, frame_w, frame_h, eye_color)

                # Track EAR history
                ear_history.append(ear)
                if len(ear_history) > 100:
                    ear_history.pop(0)

                # Check for drowsiness
                if ear < ear_threshold:
                    frame_counter += 1
                    if frame_counter >= CONSECUTIVE_FRAMES_ALERT:
                        if not alert_active:
                            alert_active = True
                            print("  ⚠ DROWSINESS ALERT!")
                        # Flash red border
                        cv2.rectangle(frame, (0, 0), (frame_w, frame_h),
                                      (0, 0, 255), 8)
                        put_text(frame, "!! DROWSINESS ALERT !!",
                                 (frame_w // 2 - 200, frame_h // 2),
                                 color=(0, 0, 255), scale=1.2, thickness=3)
                        if HAS_WINSOUND:
                            try:
                                winsound.Beep(ALERT_BEEP_FREQ, ALERT_BEEP_DURATION)
                            except Exception:
                                pass
                else:
                    if frame_counter >= 3 and not alert_active:
                        blink_counter += 1
                    frame_counter = 0
                    alert_active = False

            # Draw EAR mini graph
            graph_x, graph_y = frame_w - 220, frame_h - 120
            cv2.rectangle(frame, (graph_x - 5, graph_y - 5),
                          (graph_x + 205, graph_y + 105), (30, 30, 30), -1)
            cv2.rectangle(frame, (graph_x - 5, graph_y - 5),
                          (graph_x + 205, graph_y + 105), (100, 100, 100), 1)
            put_text(frame, "EAR Graph", (graph_x, graph_y - 10),
                     color=(200, 200, 200), scale=0.4)

            # Draw threshold line
            thresh_y = graph_y + 100 - int(ear_threshold * 250)
            cv2.line(frame, (graph_x, thresh_y), (graph_x + 200, thresh_y),
                     (0, 255, 255), 1)

            if len(ear_history) > 1:
                for i in range(1, len(ear_history)):
                    x1 = graph_x + int((i - 1) * 200 / 100)
                    x2 = graph_x + int(i * 200 / 100)
                    y1 = graph_y + 100 - int(ear_history[i - 1] * 250)
                    y2 = graph_y + 100 - int(ear_history[i] * 250)
                    y1 = max(graph_y, min(graph_y + 100, y1))
                    y2 = max(graph_y, min(graph_y + 100, y2))
                    color = (0, 255, 0) if ear_history[i] >= ear_threshold else (0, 0, 255)
                    cv2.line(frame, (x1, y1), (x2, y2), color, 1)

            # Header
            cv2.rectangle(frame, (0, 0), (frame_w, 50), (30, 30, 30), -1)
            status = "AWAKE" if ear >= ear_threshold or not face_detected else "DROWSY"
            status_color = (0, 255, 0) if status == "AWAKE" else (0, 0, 255)
            put_text(frame, f"EAR: {ear:.3f} | Thresh: {ear_threshold:.2f} | "
                            f"Blinks: {blink_counter} | Status: {status}",
                     (10, 35), color=status_color, scale=0.6)

            if not face_detected:
                put_text(frame, "No face detected",
                         (frame_w // 2 - 120, frame_h // 2),
                         color=(0, 255, 255), scale=0.8)

            cv2.imshow("Drowsiness Monitor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('+') or key == ord('='):
                ear_threshold = min(0.4, ear_threshold + 0.01)
                print(f"  Threshold: {ear_threshold:.2f}")
            elif key == ord('-'):
                ear_threshold = max(0.1, ear_threshold - 0.01)
                print(f"  Threshold: {ear_threshold:.2f}")
            elif key == ord('r'):
                blink_counter = 0
                frame_counter = 0
                ear_history.clear()
                print("  Counters reset.")

    print(f"\n[INFO] Drowsiness monitor stopped. Total blinks: {blink_counter}")
    release_and_close(cap)


if __name__ == "__main__":
    main()
