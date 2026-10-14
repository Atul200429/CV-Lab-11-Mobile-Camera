"""
camera_utils.py  –  Shared Camera Utility Module
=================================================
Provides a unified interface for connecting to:
  • IP Webcam (Android/iOS) stream over Wi-Fi
  • Local USB / built-in webcam (fallback)

Usage:
    from utils.camera_utils import get_camera

    cap = get_camera()          # uses default IP or webcam 0
    cap = get_camera(source=0)  # force local webcam
    cap = get_camera(source="http://192.168.1.5:8080/video")
"""

import cv2
import sys
import time

# ──────────────────────────────────────────────
# DEFAULT IP WEBCAM URL  (change to your phone's IP)
# Android  →  Install "IP Webcam" app, start server, use the URL shown.
# iOS      →  Install "iVCam" or "EpocCam" and use the URL from the app.
# ──────────────────────────────────────────────
DEFAULT_IP_URL = "http://192.168.1.5:8080/video"


def get_camera(source=None, width=640, height=480, retries=3):
    """
    Open a video capture from the given source.

    Parameters
    ----------
    source : str | int | None
        • None          → tries DEFAULT_IP_URL first, then falls back to webcam 0
        • int (0, 1, …) → local webcam index
        • str (URL)     → IP webcam stream URL
    width  : int – desired frame width
    height : int – desired frame height
    retries: int – number of connection attempts for IP streams

    Returns
    -------
    cv2.VideoCapture  (opened and ready)
    """
    if source is None:
        # Try IP webcam first
        cap = _try_ip_stream(DEFAULT_IP_URL, retries)
        if cap is not None:
            _set_resolution(cap, width, height)
            return cap
        print("[INFO] IP Webcam not reachable. Falling back to local webcam (index 0).")
        source = 0

    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open local webcam at index {source}.")
            sys.exit(1)
        _set_resolution(cap, width, height)
        print(f"[INFO] Local webcam {source} opened successfully.")
        return cap

    if isinstance(source, str):
        cap = _try_ip_stream(source, retries)
        if cap is None:
            print(f"[ERROR] Cannot connect to IP stream: {source}")
            sys.exit(1)
        _set_resolution(cap, width, height)
        return cap

    print("[ERROR] Invalid source type.")
    sys.exit(1)


def _try_ip_stream(url, retries=3):
    """Attempt to connect to an IP webcam stream with retries."""
    for attempt in range(1, retries + 1):
        print(f"[INFO] Connecting to IP stream: {url}  (attempt {attempt}/{retries})")
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[INFO] Connected to IP stream: {url}")
                return cap
        cap.release()
        time.sleep(1)
    return None


def _set_resolution(cap, width, height):
    """Set capture resolution (best-effort; some sources ignore this)."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def release_and_close(cap):
    """Release capture and destroy all OpenCV windows."""
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


def put_text(frame, text, position=(10, 30),
             font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.8,
             color=(0, 255, 0), thickness=2):
    """Draw text on a frame (convenience wrapper)."""
    cv2.putText(frame, text, position, font, scale, color, thickness, cv2.LINE_AA)
    return frame
