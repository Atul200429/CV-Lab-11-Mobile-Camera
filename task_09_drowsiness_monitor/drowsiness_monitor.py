"""
Task 9: Mobile Driver Drowsiness / Blink Monitor
==================================================
Tracks facial landmarks to calculate the Eye Aspect Ratio (EAR).
If the user closes their eyes for longer than a specified duration,
plays an audible alarm through the PC speakers.

Key Skills: Mediapipe Face Mesh, Euclidean distance, audio triggers.

EAR Formula:
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Where p1-p6 are the six eye landmarks.
    EAR drops close to 0 when eyes are closed.

Controls:
    q / ESC  →  Quit
    +/-      →  Adjust EAR threshold
    r        →  Reset alarm

Usage:
    python drowsiness_monitor.py
    python drowsiness_monitor.py --source 0
    python drowsiness_monitor.py --ear-threshold 0.22
"""

import cv2
import numpy as np
import os
import sys
import argparse
import time
import math

# ── Add project root to path ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.camera_utils import get_camera, release_and_close, put_text

try:
    import mediapipe as mp
except ImportError:
    print("[ERROR] mediapipe is required. Install: pip install mediapipe")
    sys.exit(1)

# Audio alarm
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

try:
    import pygame
    pygame.mixer.init()
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

# ── Mediapipe Face Mesh setup ──
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ── Eye landmark indices (Mediapipe Face Mesh) ──
# Left eye
LEFT_EYE = [362, 385, 387, 263, 373, 380]
# Right eye
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# ── Configuration ──
EAR_THRESHOLD = 0.22       # Below this = eyes closed
CLOSED_DURATION = 2.0       # Seconds of closed eyes before alarm
ALARM_FREQUENCY = 2500      # Hz
ALARM_DURATION = 500        # ms


def parse_args():
    parser = argparse.ArgumentParser(description="Drowsiness / Blink Monitor")
    parser.add_argument("--source", default=None)
    parser.add_argument("--ear-threshold", type=float, default=EAR_THRESHOLD,
                        help=f"EAR threshold (default: {EAR_THRESHOLD})")
    parser.add_argument("--closed-duration", type=float, default=CLOSED_DURATION,
                        help=f"Seconds before alarm (default: {CLOSED_DURATION})")
    return parser.parse_args()


def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0]) ** 2 +
                     (point1[1] - point2[1]) ** 2)


def calculate_ear(landmarks, eye_indices, frame_w, frame_h):
    """
    Calculate Eye Aspect Ratio for one eye.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    # Get pixel coordinates
    coords = []
    for idx in eye_indices:
        lm = landmarks[idx]
        coords.append((int(lm.x * frame_w), int(lm.y * frame_h)))

    # p1=coords[0], p2=coords[1], p3=coords[2],
    # p4=coords[3], p5=coords[4], p6=coords[5]

    # Vertical distances
    v1 = euclidean_distance(coords[1], coords[5])  # p2-p6
    v2 = euclidean_distance(coords[2], coords[4])  # p3-p5

    # Horizontal distance
    h = euclidean_distance(coords[0], coords[3])    # p1-p4

    if h == 0:
        return 0.0

    ear = (v1 + v2) / (2.0 * h)
    return ear


def draw_eye_landmarks(frame, landmarks, eye_indices, frame_w, frame_h, color):
    """Draw eye landmarks and connections on the frame."""
    coords = []
    for idx in eye_indices:
        lm = landmarks[idx]
        x, y = int(lm.x * frame_w), int(lm.y * frame_h)
        coords.append((x, y))
        cv2.circle(frame, (x, y), 2, color, -1)

    # Draw eye outline
    pts = np.array(coords, dtype=np.int32)
    cv2.polylines(frame, [pts], True, color, 1)


def play_alarm():
    """Play alarm sound."""
    if HAS_WINSOUND:
        try:
            winsound.Beep(ALARM_FREQUENCY, ALARM_DURATION)
        except Exception:
            pass
    elif HAS_PYGAME:
        try:
            # Generate a beep tone
            sample_rate = 22050
            duration_sec = ALARM_DURATION / 1000
            n_samples = int(sample_rate * duration_sec)
            buf = np.zeros((n_samples, 2), dtype=np.int16)
            t = np.linspace(0, duration_sec, n_samples, endpoint=False)
            wave = (32767 * np.sin(2 * np.pi * ALARM_FREQUENCY * t)).astype(np.int16)
            buf[:, 0] = wave
            buf[:, 1] = wave
            sound = pygame.sndarray.make_sound(buf)
            sound.play()
        except Exception:
            pass


def main():
    args = parse_args()

    source = args.source
    if source is not None:
        try:
            source = int(source)
        except ValueError:
            pass

    cap = get_camera(source=source)
    ear_threshold = args.ear_threshold
    closed_duration = args.closed_duration

    # State tracking
    eyes_closed_start = None
    alarm_active = False
    blink_count = 0
    was_closed = False
    ear_history = []
    last_alarm_time = 0

    print("=" * 60)
    print("  DROWSINESS / BLINK MONITOR")
    print("=" * 60)
    print(f"  EAR Threshold    : {ear_threshold}")
    print(f"  Alarm after      : {closed_duration}s of closed eyes")
    print(f"  Audio            : {'winsound' if HAS_WINSOUND else 'pygame' if HAS_PYGAME else 'NONE'}")
    print("  Controls         : +/- adjust threshold | r reset | q/ESC quit")
    print("=" * 60)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_h, frame_w = frame.shape[:2]

            # Convert BGR → RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            current_ear = 0.0
            face_detected = False

            if results.multi_face_landmarks:
                face_detected = True
                face_landmarks = results.multi_face_landmarks[0].landmark

                # Calculate EAR for both eyes
                left_ear = calculate_ear(face_landmarks, LEFT_EYE,
                                         frame_w, frame_h)
                right_ear = calculate_ear(face_landmarks, RIGHT_EYE,
                                          frame_w, frame_h)
                current_ear = (left_ear + right_ear) / 2.0

                # Store EAR history for graph
                ear_history.append(current_ear)
                if len(ear_history) > 100:
                    ear_history.pop(0)

                # Draw eye landmarks
                eye_color = (0, 255, 0) if current_ear >= ear_threshold else (0, 0, 255)
                draw_eye_landmarks(frame, face_landmarks, LEFT_EYE,
                                   frame_w, frame_h, eye_color)
                draw_eye_landmarks(frame, face_landmarks, RIGHT_EYE,
                                   frame_w, frame_h, eye_color)

                # Check drowsiness
                if current_ear < ear_threshold:
                    if eyes_closed_start is None:
                        eyes_closed_start = time.time()

                    closed_time = time.time() - eyes_closed_start

                    if closed_time >= closed_duration:
                        alarm_active = True

                        # Flash red border
                        cv2.rectangle(frame, (0, 0),
                                      (frame_w - 1, frame_h - 1),
                                      (0, 0, 255), 6)
                        put_text(frame, "!! DROWSINESS ALERT !!",
                                 (10, frame_h // 2),
                                 color=(0, 0, 255), scale=1.2, thickness=3)
                        put_text(frame, f"Eyes closed: {closed_time:.1f}s",
                                 (10, frame_h // 2 + 40),
                                 color=(0, 0, 255), scale=0.8)

                        # Play alarm (with cooldown)
                        if time.time() - last_alarm_time > 1.5:
                            play_alarm()
                            last_alarm_time = time.time()
                    else:
                        # Warning state
                        cv2.rectangle(frame, (0, 0),
                                      (frame_w - 1, frame_h - 1),
                                      (0, 165, 255), 3)
                        put_text(frame, f"Eyes closing... {closed_time:.1f}s",
                                 (10, 70), color=(0, 165, 255), scale=0.7)

                    was_closed = True
                else:
                    if was_closed:
                        blink_count += 1
                        was_closed = False

                    eyes_closed_start = None
                    alarm_active = False

            # ── EAR Graph (bottom right corner) ──
            graph_w, graph_h = 200, 80
            graph_x = frame_w - graph_w - 10
            graph_y = frame_h - graph_h - 10

            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (graph_x, graph_y),
                          (graph_x + graph_w, graph_y + graph_h),
                          (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # Draw threshold line
            thresh_y = int(graph_y + graph_h - (ear_threshold / 0.5) * graph_h)
            cv2.line(frame, (graph_x, thresh_y),
                     (graph_x + graph_w, thresh_y), (0, 255, 255), 1)

            # Draw EAR history
            if len(ear_history) > 1:
                for i in range(1, len(ear_history)):
                    x1 = graph_x + int((i - 1) / 100 * graph_w)
                    x2 = graph_x + int(i / 100 * graph_w)
                    y1 = int(graph_y + graph_h - (ear_history[i-1] / 0.5) * graph_h)
                    y2 = int(graph_y + graph_h - (ear_history[i] / 0.5) * graph_h)
                    y1 = max(graph_y, min(graph_y + graph_h, y1))
                    y2 = max(graph_y, min(graph_y + graph_h, y2))
                    color = (0, 255, 0) if ear_history[i] >= ear_threshold else (0, 0, 255)
                    cv2.line(frame, (x1, y1), (x2, y2), color, 1)

            cv2.putText(frame, "EAR", (graph_x + 5, graph_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # ── Header ──
            cv2.rectangle(frame, (0, 0), (frame_w, 50), (30, 30, 30), -1)
            status = "ALERT!" if alarm_active else ("Monitoring" if face_detected else "No face")
            status_color = (0, 0, 255) if alarm_active else (0, 255, 0)

            put_text(frame, f"Drowsiness Monitor | {status}",
                     (10, 22), color=status_color, scale=0.6)
            put_text(frame, f"EAR: {current_ear:.3f} | Threshold: {ear_threshold:.2f} | "
                          f"Blinks: {blink_count}",
                     (10, 42), color=(200, 200, 200), scale=0.45)

            cv2.imshow("Drowsiness Monitor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('+') or key == ord('='):
                ear_threshold = min(0.40, ear_threshold + 0.01)
                print(f"[INFO] EAR threshold: {ear_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                ear_threshold = max(0.10, ear_threshold - 0.01)
                print(f"[INFO] EAR threshold: {ear_threshold:.2f}")
            elif key == ord('r'):
                alarm_active = False
                eyes_closed_start = None
                blink_count = 0
                ear_history.clear()
                print("[INFO] Monitor reset.")

    print(f"\n[INFO] Drowsiness monitor stopped. Total blinks: {blink_count}")
    release_and_close(cap)


if __name__ == "__main__":
    main()
