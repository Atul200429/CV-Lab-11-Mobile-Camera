"""
Task 5: Document Scanner with Perspective Transform
====================================================
Uses the camera to photograph documents lying on a table.
Automatically detects the four corners of the sheet, applies a
perspective warp to flatten it, and converts to a high-contrast
black-and-white scan.

Key Skills: Canny edge detection, cv2.findContours(),
            cv2.getPerspectiveTransform(), adaptive thresholding.

Controls:
    q / ESC  →  Quit
    SPACE    →  Capture & save the scanned document
    t        →  Toggle between color scan and B/W scan
    +/-      →  Adjust Canny threshold

Usage:
    python document_scanner.py
    python document_scanner.py --source 0
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

# ── Configuration ──
SCAN_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "scanned_docs")
CANNY_LOW = 50
CANNY_HIGH = 150
BLUR_KERNEL = (5, 5)
MIN_AREA_RATIO = 0.1  # Document must be at least 10% of frame area


def parse_args():
    parser = argparse.ArgumentParser(description="Document Scanner")
    parser.add_argument("--source", default=None)
    return parser.parse_args()


def order_points(pts):
    """
    Order four points as: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Sum of coordinates: smallest = top-left, largest = bottom-right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Difference: smallest = top-right, largest = bottom-left
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    return rect


def four_point_transform(image, pts):
    """
    Apply perspective transform to obtain a top-down view.
    """
    rect = order_points(pts)
    tl, tr, br, bl = rect

    # Compute width
    width_bottom = np.linalg.norm(br - bl)
    width_top = np.linalg.norm(tr - tl)
    max_width = max(int(width_bottom), int(width_top))

    # Compute height
    height_right = np.linalg.norm(tr - br)
    height_left = np.linalg.norm(tl - bl)
    max_height = max(int(height_right), int(height_left))

    # Destination points
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # Perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped


def find_document_contour(frame, canny_low, canny_high):
    """
    Find the largest four-sided contour in the frame (the document).
    Returns the 4-point contour or None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
    edged = cv2.Canny(blurred, canny_low, canny_high)

    # Dilate to close gaps in edges
    kernel = np.ones((5, 5), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)
    edged = cv2.erode(edged, kernel, iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    frame_area = frame.shape[0] * frame.shape[1]
    doc_contour = None

    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Look for a 4-point contour of sufficient size
        area = cv2.contourArea(approx)
        if len(approx) == 4 and area > frame_area * MIN_AREA_RATIO:
            doc_contour = approx
            break

    return doc_contour, edged


def apply_bw_scan(image):
    """Convert warped document to high-contrast B/W scan."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Adaptive thresholding for clean B/W output
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 10
    )
    return bw


def main():
    args = parse_args()

    source = args.source
    if source is not None:
        try:
            source = int(source)
        except ValueError:
            pass

    os.makedirs(SCAN_OUTPUT_DIR, exist_ok=True)

    cap = get_camera(source=source)
    canny_low = CANNY_LOW
    canny_high = CANNY_HIGH
    show_bw = True
    scan_count = 0

    print("=" * 60)
    print("  DOCUMENT SCANNER")
    print("=" * 60)
    print(f"  Output dir : {os.path.abspath(SCAN_OUTPUT_DIR)}")
    print("  Controls   : SPACE to capture | t toggle B/W | +/- Canny threshold")
    print("               q/ESC to quit")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()

        # Find document
        doc_contour, edged = find_document_contour(frame, canny_low, canny_high)

        scanned = None

        if doc_contour is not None:
            # Draw detected document outline (green)
            cv2.drawContours(display, [doc_contour], -1, (0, 255, 0), 3)

            # Draw corner points
            for point in doc_contour.reshape(4, 2):
                cv2.circle(display, tuple(point), 8, (0, 0, 255), -1)

            # Apply perspective transform
            pts = doc_contour.reshape(4, 2).astype("float32")
            scanned = four_point_transform(frame, pts)

            if show_bw:
                scanned_display = apply_bw_scan(scanned)
                scanned_display = cv2.cvtColor(scanned_display, cv2.COLOR_GRAY2BGR)
            else:
                scanned_display = scanned.copy()

            # Resize scanned preview to fit
            h, w = scanned_display.shape[:2]
            max_h = 500
            if h > max_h:
                scale = max_h / h
                scanned_display = cv2.resize(scanned_display, None,
                                             fx=scale, fy=scale)

            cv2.imshow("Scanned Document", scanned_display)

            put_text(display, "Document detected! Press SPACE to save",
                     (10, display.shape[0] - 20), color=(0, 255, 0), scale=0.6)
        else:
            put_text(display, "No document detected - adjust position",
                     (10, display.shape[0] - 20), color=(0, 0, 255), scale=0.6)

        # Header
        mode = "B/W" if show_bw else "Color"
        cv2.rectangle(display, (0, 0), (display.shape[1], 40), (30, 30, 30), -1)
        put_text(display, f"Doc Scanner | Mode: {mode} | "
                          f"Canny: {canny_low}/{canny_high} | Saved: {scan_count}",
                 (10, 28), color=(0, 255, 0), scale=0.55)

        cv2.imshow("Document Scanner - Live", display)

        # Show edge detection
        edged_color = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Edge Detection", edged_color)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' ') and scanned is not None:
            # Save scan
            scan_count += 1
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if show_bw:
                bw_scan = apply_bw_scan(scanned)
                filename = f"scan_{timestamp}_bw.png"
                cv2.imwrite(os.path.join(SCAN_OUTPUT_DIR, filename), bw_scan)
            else:
                filename = f"scan_{timestamp}_color.png"
                cv2.imwrite(os.path.join(SCAN_OUTPUT_DIR, filename), scanned)
            print(f"  ✔ Scan saved: {filename}")
        elif key == ord('t'):
            show_bw = not show_bw
            print(f"[INFO] Mode: {'B/W' if show_bw else 'Color'}")
        elif key == ord('+') or key == ord('='):
            canny_low = min(200, canny_low + 10)
            canny_high = min(300, canny_high + 10)
            print(f"[INFO] Canny: {canny_low}/{canny_high}")
        elif key == ord('-') or key == ord('_'):
            canny_low = max(10, canny_low - 10)
            canny_high = max(50, canny_high - 10)
            print(f"[INFO] Canny: {canny_low}/{canny_high}")

    print(f"\n[INFO] Document scanner stopped. {scan_count} scan(s) saved.")
    release_and_close(cap)


if __name__ == "__main__":
    main()
