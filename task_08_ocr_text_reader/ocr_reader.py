"""
Task 8: Real-Time Optical Character Recognition (OCR) Text Reader
==================================================================
Aims the camera at printed text (book pages, signs, product labels).
Extracts text live and displays the recognized string as a dynamic
overlay on the video feed.

Key Skills: Pytesseract (Tesseract OCR API), image thresholding,
            text overlay (cv2.putText).

Prerequisites:
    1. Install Tesseract OCR: https://github.com/tesseract-ocr/tesseract
       Windows: download installer from https://github.com/UB-Mannheim/tesseract/wiki
    2. pip install pytesseract

Controls:
    q / ESC  →  Quit
    SPACE    →  Freeze frame and run full OCR
    r        →  Resume live feed
    s        →  Save recognized text to file
    t        →  Toggle preprocessing mode

Usage:
    python ocr_reader.py
    python ocr_reader.py --source 0
    python ocr_reader.py --lang eng+hin    # multiple languages
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

try:
    import pytesseract
except ImportError:
    print("[ERROR] pytesseract is required. Install: pip install pytesseract")
    sys.exit(1)

# ── Configure Tesseract path (Windows) ──
# Adjust this path if Tesseract is installed elsewhere
TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\Atul Krishna\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
]

for path in TESSERACT_PATHS:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        break

# ── Configuration ──
OCR_INTERVAL_FRAMES = 15  # Run OCR every N frames (for performance)
PREPROCESS_MODES = ["original", "grayscale", "threshold", "adaptive", "blur+threshold"]


def parse_args():
    parser = argparse.ArgumentParser(description="OCR Text Reader")
    parser.add_argument("--source", default=None)
    parser.add_argument("--lang", default="eng",
                        help="Tesseract language(s), e.g., eng, eng+hin")
    return parser.parse_args()


def preprocess_for_ocr(frame, mode):
    """Apply preprocessing to improve OCR accuracy."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if mode == "original":
        return frame
    elif mode == "grayscale":
        return gray
    elif mode == "threshold":
        _, thresh = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    elif mode == "adaptive":
        thresh = cv2.adaptiveThreshold(gray, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        return thresh
    elif mode == "blur+threshold":
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    return frame


def draw_ocr_boxes(frame, ocr_data):
    """Draw bounding boxes around detected text."""
    n_boxes = len(ocr_data['text'])
    for i in range(n_boxes):
        text = ocr_data['text'][i].strip()
        conf = int(ocr_data['conf'][i])

        if conf > 40 and text:  # Filter low-confidence detections
            x = ocr_data['left'][i]
            y = ocr_data['top'][i]
            w = ocr_data['width'][i]
            h = ocr_data['height'][i]

            # Color based on confidence
            if conf > 80:
                color = (0, 255, 0)   # Green = high confidence
            elif conf > 60:
                color = (0, 255, 255) # Yellow = medium
            else:
                color = (0, 165, 255) # Orange = low

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
            cv2.putText(frame, f"{text} ({conf}%)",
                        (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, color, 1, cv2.LINE_AA)


def wrap_text(text, max_width, font, scale, thickness):
    """Split text into lines that fit within max_width pixels."""
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        (w, _), _ = cv2.getTextSize(test_line, font, scale, thickness)
        if w <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines


def main():
    args = parse_args()

    source = args.source
    if source is not None:
        try:
            source = int(source)
        except ValueError:
            pass

    cap = get_camera(source=source)

    # Verify Tesseract installation
    try:
        version = pytesseract.get_tesseract_version()
        print(f"[INFO] Tesseract version: {version}")
    except Exception:
        print("[ERROR] Tesseract OCR not found!")
        print("        Install from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("        Then set the path in the script or add to system PATH.")
        release_and_close(cap)
        sys.exit(1)

    preprocess_idx = 0
    frame_count = 0
    recognized_text = ""
    frozen = False
    frozen_frame = None
    ocr_data = None
    save_dir = os.path.join(os.path.dirname(__file__), "..", "snapshots")
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("  REAL-TIME OCR TEXT READER")
    print("=" * 60)
    print(f"  Language    : {args.lang}")
    print(f"  Preprocess  : {PREPROCESS_MODES[preprocess_idx]}")
    print("  Controls    : SPACE=freeze+OCR | r=resume | t=toggle mode")
    print("                s=save text | q/ESC quit")
    print("=" * 60)

    while True:
        if not frozen:
            ret, frame = cap.read()
            if not ret:
                continue
        else:
            frame = frozen_frame.copy()

        display = frame.copy()
        frame_count += 1

        # Preprocess
        mode = PREPROCESS_MODES[preprocess_idx]
        processed = preprocess_for_ocr(frame, mode)

        # Run OCR periodically (or when frozen)
        if frozen or (frame_count % OCR_INTERVAL_FRAMES == 0):
            try:
                # Get text
                recognized_text = pytesseract.image_to_string(
                    processed, lang=args.lang
                ).strip()

                # Get bounding box data
                ocr_data = pytesseract.image_to_data(
                    processed, lang=args.lang,
                    output_type=pytesseract.Output.DICT
                )
            except Exception as e:
                recognized_text = f"[OCR Error: {e}]"
                ocr_data = None

        # Draw bounding boxes
        if ocr_data:
            draw_ocr_boxes(display, ocr_data)

        # Text overlay panel at bottom
        if recognized_text:
            lines = recognized_text.split('\n')[:6]  # Show first 6 lines
            panel_h = len(lines) * 22 + 10
            panel_y = display.shape[0] - panel_h

            # Semi-transparent background
            overlay = display.copy()
            cv2.rectangle(overlay, (0, panel_y),
                          (display.shape[1], display.shape[0]),
                          (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

            for i, line in enumerate(lines):
                if line.strip():
                    cv2.putText(display, line.strip()[:80],
                                (10, panel_y + 20 + i * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 255, 255), 1, cv2.LINE_AA)

        # Header
        status = "FROZEN - Full OCR" if frozen else "Live"
        cv2.rectangle(display, (0, 0), (display.shape[1], 40), (30, 30, 30), -1)
        put_text(display, f"OCR Reader | {status} | Mode: {mode}",
                 (10, 28), color=(0, 255, 0), scale=0.55)

        # Show preprocessed view
        if len(processed.shape) == 2:
            processed_display = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        else:
            processed_display = processed

        cv2.imshow("OCR Text Reader", display)
        cv2.imshow("Preprocessed View", processed_display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            frozen = True
            frozen_frame = frame.copy()
            print("[INFO] Frame frozen. Running full OCR...")
        elif key == ord('r'):
            frozen = False
            print("[INFO] Resumed live feed.")
        elif key == ord('t'):
            preprocess_idx = (preprocess_idx + 1) % len(PREPROCESS_MODES)
            print(f"[INFO] Preprocess mode: {PREPROCESS_MODES[preprocess_idx]}")
        elif key == ord('s'):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            text_file = os.path.join(save_dir, f"ocr_text_{timestamp}.txt")
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(recognized_text)
            cv2.imwrite(os.path.join(save_dir, f"ocr_frame_{timestamp}.png"), frame)
            print(f"  ✔ Text saved: {text_file}")

    print(f"\n[INFO] OCR reader stopped.")
    if recognized_text:
        print(f"\nLast recognized text:\n{'-'*40}\n{recognized_text}\n{'-'*40}")
    release_and_close(cap)


if __name__ == "__main__":
    main()
