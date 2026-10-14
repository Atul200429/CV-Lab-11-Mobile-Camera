"""
Task 3: Hand Gesture Desktop Controller
========================================
Detects hand landmarks using Mediapipe Hands and translates finger
count or gestures into system actions:
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

# ── Add project root to path ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.camera_utils import get_camera, release_and_close, put_text

try:
    import mediapipe as mp
except ImportError:
    print("[ERROR] mediapipe is required. Install: pip install mediapipe")
    sys.exit(1)

try:
    import pyautogui
    pyautogui.FAILSAFE = False
except ImportError:
    print("[ERROR] pyautogui is required. Install: pip install pyautogui")
    sys.exit(1)

# ── Mediapipe setup ──
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ── Configuration ──
ACTION_COOLDOWN = 0.8  # seconds between actions to prevent spam
CONFIDENCE_THRESHOLD = 0.7

# Finger tip and pip landmark indices
FINGER_TIPS = [8, 12, 16, 20]   # Index, Middle, Ring, Pinky tips
FINGER_PIPS = [6, 10, 14, 18]   # Corresponding PIP joints
THUMB_TIP = 4
THUMB_IP = 3
THUMB_MCP = 2


def parse_args():
    parser = argparse.ArgumentParser(description="Hand Gesture Desktop Controller")
    parser.add_argument("--source", default=None,
                        help="Camera source: int for webcam index, URL for IP stream")
    return parser.parse_args()


def count_fingers(hand_landmarks, handedness):
    """Count the number of extended fingers."""
    landmarks = hand_landmarks.landmark
    fingers = []

    # Thumb: compare tip x-position with IP joint
    # For right hand: tip should be to the LEFT of IP (lower x) when extended
    # For left hand: tip should be to the RIGHT of IP (higher x) when extended
    is_right = handedness.classification[0].label == "Right"

    if is_right:
        # In mirrored view, "Right" hand appears on left side
        thumb_extended = landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x
    else:
        thumb_extended = landmarks[THUMB_TIP].x > landmarks[THUMB_IP].x

    fingers.append(thumb_extended)

    # Other four fingers: tip should be above (lower y) PIP when extended
    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        fingers.append(landmarks[tip].y < landmarks[pip].y)

    return fingers, sum(fingers)


def get_gesture_action(finger_states, count):
    """Map finger count / state to an action."""
    thumb, index, middle, ring, pinky = finger_states

    if count == 0:
        return "SCROLL DOWN", "Fist → Scroll Down"
    elif count == 1 and index:
        return "VOLUME UP", "Index → Volume Up"
    elif count == 1 and thumb:
        return "SCROLL UP", "Thumb → Scroll Up"
    elif count == 2 and index and middle:
        return "VOLUME DOWN", "Peace → Volume Down"
    elif count == 3:
        return "MUTE", "Three → Mute Toggle"
    elif count == 4:
        return "NEXT SLIDE", "Four → Next Slide"
    elif count == 5:
        return "PREV SLIDE", "Open Palm → Prev Slide"
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


def main():
    args = parse_args()

    source = args.source
    if source is not None:
        try:
            source = int(source)
        except ValueError:
            pass

    cap = get_camera(source=source)
    last_action_time = 0

    print("=" * 60)
    print("  HAND GESTURE DESKTOP CONTROLLER")
    print("=" * 60)
    print("  Gestures:")
    print("    1 finger (Index)     → Volume Up")
    print("    2 fingers (Peace)    → Volume Down")
    print("    3 fingers            → Mute Toggle")
    print("    4 fingers            → Next Slide (→)")
    print("    5 fingers (Palm)     → Prev Slide (←)")
    print("    Fist (0 fingers)     → Scroll Down")
    print("    Thumb only           → Scroll Up")
    print("  Controls: q/ESC to quit")
    print("=" * 60)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=CONFIDENCE_THRESHOLD,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Flip for mirror view
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB for Mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            gesture_text = "No hand detected"
            action_text = ""

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Count fingers
                    finger_states, count = count_fingers(hand_landmarks, handedness)
                    action, gesture_text = get_gesture_action(finger_states, count)

                    # Execute action with cooldown
                    current_time = time.time()
                    if action and (current_time - last_action_time) > ACTION_COOLDOWN:
                        execute_action(action)
                        action_text = f">> {action}"
                        last_action_time = current_time
                        print(f"  ✔ {gesture_text} → {action}")

                    # Draw finger count circle
                    cv2.circle(frame, (frame.shape[1] - 60, 60), 45, (0, 200, 0), -1)
                    cv2.putText(frame, str(count),
                                (frame.shape[1] - 75, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                (255, 255, 255), 3, cv2.LINE_AA)

            # UI overlay
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (30, 30, 30), -1)
            put_text(frame, f"Gesture: {gesture_text}", (10, 30),
                     color=(0, 255, 0), scale=0.7)

            if action_text:
                put_text(frame, action_text,
                         (10, frame.shape[0] - 20),
                         color=(0, 255, 255), scale=0.8)

            cv2.imshow("Hand Gesture Controller", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

    print("\n[INFO] Gesture controller stopped.")
    release_and_close(cap)


if __name__ == "__main__":
    main()
