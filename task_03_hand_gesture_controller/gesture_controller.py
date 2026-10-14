"""
Task 3: Hand Gesture Desktop Controller
========================================
Detects hand landmarks using Mediapipe HandLandmarker and translates
finger count or gestures into system actions:
    • 1 finger  (Index)         → Volume UP
    • 2 fingers (Index+Middle)  → Volume DOWN
    • 3 fingers                 → Mute/Unmute toggle
    • 4 fingers                 → Next slide (Right arrow)
    • 5 fingers (Open palm)     → Previous slide (Left arrow)
    • Fist (0 fingers)          → Scroll Down
    • Thumb only                → Scroll Up

Key Skills: Mediapipe Hands, gesture mapping, pyautogui.

Controls:
    q / ESC  →  Quit

Usage:
    python gesture_controller.py
    python gesture_controller.py --source 0
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
    HandLandmarker, HandLandmarkerOptions, HandLandmarkerResult, RunningMode
)

try:
    import pyautogui
    pyautogui.FAILSAFE = False
except ImportError:
    print("[ERROR] pyautogui is required. Install: pip install pyautogui")
    sys.exit(1)

# ── Configuration ──
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "hand_landmarker.task")
ACTION_COOLDOWN = 0.8  # seconds between actions to prevent spam

# Finger tip and pip landmark indices
FINGER_TIPS = [8, 12, 16, 20]   # Index, Middle, Ring, Pinky tips
FINGER_PIPS = [6, 10, 14, 18]   # Corresponding PIP joints
THUMB_TIP = 4
THUMB_IP = 3


def parse_args():
    parser = argparse.ArgumentParser(description="Hand Gesture Desktop Controller")
    parser.add_argument("--source", default=None,
                        help="Camera source: int for webcam index, URL for IP stream")
    return parser.parse_args()


def count_fingers(landmarks, handedness_label):
    """Count the number of extended fingers."""
    fingers = []

    is_right = handedness_label == "Right"

    # Thumb
    if is_right:
        thumb_extended = landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x
    else:
        thumb_extended = landmarks[THUMB_TIP].x > landmarks[THUMB_IP].x

    fingers.append(thumb_extended)

    # Other four fingers
    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        fingers.append(landmarks[tip].y < landmarks[pip].y)

    return fingers, sum(fingers)


def get_gesture_action(finger_states, count):
    """Map finger count / state to an action."""
    thumb, index, middle, ring, pinky = finger_states

    if count == 0:
        return "SCROLL DOWN", "Fist -> Scroll Down"
    elif count == 1 and index:
        return "VOLUME UP", "Index -> Volume Up"
    elif count == 1 and thumb:
        return "SCROLL UP", "Thumb -> Scroll Up"
    elif count == 2 and index and middle:
        return "VOLUME DOWN", "Peace -> Volume Down"
    elif count == 3:
        return "MUTE", "Three -> Mute Toggle"
    elif count == 4:
        return "NEXT SLIDE", "Four -> Next Slide"
    elif count == 5:
        return "PREV SLIDE", "Open Palm -> Prev Slide"
    else:
        return None, f"Fingers: {count}"


def execute_action(action):
    """Execute the mapped system action."""
    if action == "VOLUME UP":
        pyautogui.press("volumeup")
    elif action == "VOLUME DOWN":
        pyautogui.press("volumedown")
    elif action == "MUTE":
        pyautogui.press("volumemute")
    elif action == "NEXT SLIDE":
        pyautogui.press("right")
    elif action == "PREV SLIDE":
        pyautogui.press("left")
    elif action == "SCROLL UP":
        pyautogui.scroll(3)
    elif action == "SCROLL DOWN":
        pyautogui.scroll(-3)


def draw_hand_landmarks(frame, landmarks, w, h, color=(0, 255, 0)):
    """Draw hand landmarks and connections on frame."""
    # Draw landmarks
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, color, -1)

    # Draw connections
    connections = [
        (0,1),(1,2),(2,3),(3,4),  # Thumb
        (0,5),(5,6),(6,7),(7,8),  # Index
        (5,9),(9,10),(10,11),(11,12),  # Middle
        (9,13),(13,14),(14,15),(15,16),  # Ring
        (13,17),(17,18),(18,19),(19,20),  # Pinky
        (0,17)  # Palm
    ]
    for start, end in connections:
        x1, y1 = int(landmarks[start].x * w), int(landmarks[start].y * h)
        x2, y2 = int(landmarks[end].x * w), int(landmarks[end].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), color, 2)


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
              "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        sys.exit(1)

    cap = get_camera(source=source)
    last_action_time = 0

    # Store latest result from callback
    latest_result = [None]

    def result_callback(result, output_image, timestamp_ms):
        latest_result[0] = result

    # Create HandLandmarker
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=result_callback,
    )

    print("=" * 60)
    print("  HAND GESTURE DESKTOP CONTROLLER")
    print("=" * 60)
    print("  Gestures:")
    print("    1 finger (Index)     -> Volume Up")
    print("    2 fingers (Peace)    -> Volume Down")
    print("    3 fingers            -> Mute Toggle")
    print("    4 fingers            -> Next Slide")
    print("    5 fingers (Palm)     -> Prev Slide")
    print("    Fist (0 fingers)     -> Scroll Down")
    print("    Thumb only           -> Scroll Up")
    print("  Controls: q/ESC to quit")
    print("=" * 60)

    with HandLandmarker.create_from_options(options) as landmarker:
        frame_timestamp = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]

            # Convert to mediapipe Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            frame_timestamp += 33  # ~30 fps
            landmarker.detect_async(mp_image, frame_timestamp)

            gesture_text = "No hand detected"
            action_text = ""

            result = latest_result[0]
            if result and result.hand_landmarks:
                for i, hand_landmarks in enumerate(result.hand_landmarks):
                    handedness_label = result.handedness[i][0].category_name

                    # Draw landmarks
                    draw_hand_landmarks(frame, hand_landmarks, frame_w, frame_h)

                    # Count fingers
                    finger_states, count = count_fingers(hand_landmarks, handedness_label)
                    action, gesture_text = get_gesture_action(finger_states, count)

                    # Execute action with cooldown
                    current_time = time.time()
                    if action and (current_time - last_action_time) > ACTION_COOLDOWN:
                        execute_action(action)
                        action_text = f">> {action}"
                        last_action_time = current_time
                        print(f"  -> {gesture_text} -> {action}")

                    # Draw finger count circle
                    cv2.circle(frame, (frame_w - 60, 60), 45, (0, 200, 0), -1)
                    cv2.putText(frame, str(count),
                                (frame_w - 75, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                (255, 255, 255), 3, cv2.LINE_AA)

            # UI overlay
            cv2.rectangle(frame, (0, 0), (frame_w, 45), (30, 30, 30), -1)
            put_text(frame, f"Gesture: {gesture_text}", (10, 30),
                     color=(0, 255, 0), scale=0.7)

            if action_text:
                put_text(frame, action_text,
                         (10, frame_h - 20),
                         color=(0, 255, 255), scale=0.8)

            cv2.imshow("Hand Gesture Controller", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

    print("\n[INFO] Gesture controller stopped.")
    release_and_close(cap)


if __name__ == "__main__":
    main()
