# 📱 CV Lab 11: Real-Time Mobile Camera Integration via Python & OpenCV

> **Course:** Computer Vision Lab  
> **Objective:** Stream live video from a mobile phone camera to a PC, process the incoming feed using Python and OpenCV, and apply real-time computer vision techniques.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites & Setup](#prerequisites--setup)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Mobile Camera Setup](#mobile-camera-setup)
- [Tasks](#tasks)
  - [Task 1: Motion & Security Alarm Detector](#task-1-motion--security-alarm-detector)
  - [Task 2: QR Code & Barcode Scanner](#task-2-qr-code--barcode-scanner)
  - [Task 3: Hand Gesture Desktop Controller](#task-3-hand-gesture-desktop-controller)
  - [Task 4: Smart Attendance System](#task-4-smart-attendance-system)
  - [Task 5: Document Scanner](#task-5-document-scanner)
  - [Task 6: Color & Defect Inspector](#task-6-color--defect-inspector)
  - [Task 7: Virtual Whiteboard / Air Painting](#task-7-virtual-whiteboard--air-painting)
  - [Task 8: OCR Text Reader](#task-8-ocr-text-reader)
  - [Task 9: Drowsiness / Blink Monitor](#task-9-drowsiness--blink-monitor)
  - [Task 10: AR Filter Overlay](#task-10-ar-filter-overlay)
- [Key Concepts Summary](#key-concepts-summary)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

This repository contains **10 complete, runnable Python programs** that demonstrate real-time computer vision techniques using a mobile phone camera as the video source. Each task is self-contained, well-documented, and uses a shared camera utility module for seamless IP webcam/local webcam integration.

---

## Prerequisites & Setup

### Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.8+ | Runtime |
| OpenCV | 4.8+ | Computer vision |
| Mediapipe | 0.10+ | Hand/face landmark detection |
| Tesseract OCR | 5.0+ | Optical character recognition |
| IP Webcam App | Latest | Mobile camera streaming |

### Hardware Requirements

- PC with Python installed
- Smartphone (Android/iOS)
- Both devices on the **same Wi-Fi network**

---

## Project Structure

```
CV-Lab-11-Mobile-Camera/
│
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
│
├── utils/
│   ├── __init__.py
│   └── camera_utils.py                # Shared camera connection module
│
├── task_01_motion_security_alarm/
│   └── motion_detector.py             # Frame differencing + alarm
│
├── task_02_qr_barcode_scanner/
│   └── qr_barcode_scanner.py          # QR/Barcode → CSV logging
│
├── task_03_hand_gesture_controller/
│   └── gesture_controller.py          # Mediapipe hand → system actions
│
├── task_04_smart_attendance/
│   └── attendance_system.py           # Face recognition → attendance CSV
│
├── task_05_document_scanner/
│   └── document_scanner.py            # Perspective transform → B/W scan
│
├── task_06_color_defect_inspector/
│   └── color_inspector.py             # HSV color thresholding + analysis
│
├── task_07_virtual_whiteboard/
│   └── air_painting.py                # Object tracking → drawing canvas
│
├── task_08_ocr_text_reader/
│   └── ocr_reader.py                  # Tesseract OCR → text overlay
│
├── task_09_drowsiness_monitor/
│   └── drowsiness_monitor.py          # EAR calculation → audible alarm
│
├── task_10_ar_filter_overlay/
│   └── ar_filter.py                   # Face mesh → AR filters
│
├── known_faces/                        # (Task 4) Place known face images here
├── snapshots/                          # Auto-saved snapshots and captures
├── scanned_docs/                       # Auto-saved document scans
├── inventory_log/                      # Auto-saved CSV logs
└── assets/                             # Additional assets
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/CV-Lab-11-Mobile-Camera.git
cd CV-Lab-11-Mobile-Camera
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Additional Setup

**For Task 4 (Face Recognition):**
- Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (required for dlib)
- Or use pre-compiled wheels: `pip install dlib`

**For Task 8 (OCR):**
- Download and install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) for Windows
- Add to PATH or update the path in `ocr_reader.py`

---

## Mobile Camera Setup

### Option A: IP Webcam (Android) — Recommended

1. Install **IP Webcam** from Google Play Store
2. Open the app → scroll to bottom → tap **"Start Server"**
3. Note the URL shown (e.g., `http://192.168.1.5:8080`)
4. Update `DEFAULT_IP_URL` in `utils/camera_utils.py`:
   ```python
   DEFAULT_IP_URL = "http://YOUR_PHONE_IP:8080/video"
   ```

### Option B: iVCam / EpocCam (iOS/Android)

1. Install the app on your phone and PC client
2. Connect both to the same Wi-Fi
3. Use the stream URL provided by the app

### Option C: Local Webcam (Fallback)

All scripts automatically fall back to the local webcam (index 0) if the IP stream is unreachable. You can also force it:

```bash
python script_name.py --source 0
```

---

## Tasks

### Task 1: Motion & Security Alarm Detector

**File:** `task_01_motion_security_alarm/motion_detector.py`

**Objective:** Track frame differences in real time. If motion above a threshold is detected, trigger a visual alert and save an emergency snapshot.

**Key Concepts:**
- `cv2.absdiff()` for frame subtraction
- Gaussian blur for noise reduction
- Contour area thresholding
- Automatic snapshot saving with cooldown

**Run:**
```bash
python task_01_motion_security_alarm/motion_detector.py
python task_01_motion_security_alarm/motion_detector.py --source 0 --threshold 3000
```

**Controls:** `+/-` adjust threshold | `q/ESC` quit

---

### Task 2: QR Code & Barcode Scanner

**File:** `task_02_qr_barcode_scanner/qr_barcode_scanner.py`

**Objective:** Read QR codes and barcodes in real time, decode data, and log to a CSV file with timestamps.

**Key Concepts:**
- `pyzbar` for multi-format decoding (QR, EAN, Code128, etc.)
- OpenCV `QRCodeDetector` as fallback
- CSV logging with duplicate cooldown

**Run:**
```bash
python task_02_qr_barcode_scanner/qr_barcode_scanner.py
```

**Controls:** `c` clear log | `s` show CSV path | `q/ESC` quit

---

### Task 3: Hand Gesture Desktop Controller

**File:** `task_03_hand_gesture_controller/gesture_controller.py`

**Objective:** Detect hand landmarks and translate finger count into system actions (volume, slides, scrolling).

**Gesture Map:**

| Fingers | Gesture | Action |
|---------|---------|--------|
| 1 (Index) | Point Up | Volume Up |
| 2 (Peace) | Peace Sign | Volume Down |
| 3 | Three | Mute Toggle |
| 4 | Four | Next Slide (→) |
| 5 (Palm) | Open Palm | Prev Slide (←) |
| 0 (Fist) | Fist | Scroll Down |
| Thumb only | Thumbs Up | Scroll Up |

**Key Concepts:**
- Mediapipe Hands (21 landmarks)
- Finger extension detection via landmark positions
- `pyautogui` for system-level actions

**Run:**
```bash
python task_03_hand_gesture_controller/gesture_controller.py
```

---

### Task 4: Smart Attendance System

**File:** `task_04_smart_attendance/attendance_system.py`

**Objective:** Detect and recognize faces, match against known database, mark attendance in CSV.

**Setup:**
1. Add face photos to `known_faces/` directory
2. Name files as `Person_Name.jpg` (underscores → spaces)
3. Multiple photos: `Person_Name_1.jpg`, `Person_Name_2.jpg`

**Key Concepts:**
- `face_recognition` library with 128-D face embeddings
- Tolerance-based matching (configurable)
- Daily CSV attendance log

**Run:**
```bash
python task_04_smart_attendance/attendance_system.py
python task_04_smart_attendance/attendance_system.py --tolerance 0.45
```

**Controls:** `r` reload faces | `a` show attendance | `q/ESC` quit

---

### Task 5: Document Scanner

**File:** `task_05_document_scanner/document_scanner.py`

**Objective:** Detect document corners, apply perspective warp, convert to high-contrast B/W scan.

**Key Concepts:**
- Canny edge detection
- `cv2.findContours()` → 4-point approximation
- `cv2.getPerspectiveTransform()` + `cv2.warpPerspective()`
- Adaptive thresholding for clean B/W output

**Run:**
```bash
python task_05_document_scanner/document_scanner.py
```

**Controls:** `SPACE` capture | `t` toggle B/W | `+/-` Canny threshold | `q/ESC` quit

---

### Task 6: Color & Defect Inspector

**File:** `task_06_color_defect_inspector/color_inspector.py`

**Objective:** Isolate specific color ranges in HSV to detect objects by color (ripe/unripe fruit, defective parts).

**Presets:**

| Key | Color | Use Case |
|-----|-------|----------|
| 1 | Red | Ripe fruit detection |
| 2 | Green | Unripe fruit detection |
| 3 | Blue | Defect markers |
| 4 | Yellow | Caution zones |
| 5 | Orange | General detection |
| 6 | Purple | Special items |

**Key Concepts:**
- `cv2.cvtColor()` BGR → HSV
- `cv2.inRange()` for color thresholding
- Morphological operations (open/close)
- Contour area and coverage analysis

**Run:**
```bash
python task_06_color_defect_inspector/color_inspector.py
```

---

### Task 7: Virtual Whiteboard / Air Painting

**File:** `task_07_virtual_whiteboard/air_painting.py`

**Objective:** Track a colored object and draw its path on a virtual canvas.

**Key Concepts:**
- HSV object tracking via `cv2.inRange()`
- `cv2.moments()` for centroid calculation
- `cv2.line()` drawing on canvas buffer
- Multi-color drawing with eraser

**Run:**
```bash
python task_07_virtual_whiteboard/air_painting.py --tracker blue
python task_07_virtual_whiteboard/air_painting.py --tracker red
```

**Controls:** `1-4` color | `e` eraser | `c` clear | `u` undo | `s` save | `q/ESC` quit

---

### Task 8: OCR Text Reader

**File:** `task_08_ocr_text_reader/ocr_reader.py`

**Objective:** Extract printed text from the camera feed in real time using Tesseract OCR.

**Preprocessing Modes:**
- Original → Grayscale → Binary Threshold → Adaptive → Blur+Threshold

**Key Concepts:**
- `pytesseract.image_to_string()` for text extraction
- `pytesseract.image_to_data()` for bounding box detection
- Multiple preprocessing pipelines
- Confidence-based color coding

**Run:**
```bash
python task_08_ocr_text_reader/ocr_reader.py
python task_08_ocr_text_reader/ocr_reader.py --lang eng+hin
```

**Controls:** `SPACE` freeze+OCR | `r` resume | `t` toggle preprocessing | `s` save | `q/ESC` quit

---

### Task 9: Drowsiness / Blink Monitor

**File:** `task_09_drowsiness_monitor/drowsiness_monitor.py`

**Objective:** Calculate Eye Aspect Ratio (EAR) from facial landmarks. Trigger alarm if eyes are closed too long.

**EAR Formula:**
```
EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
```
- EAR ≈ 0.3 when eyes are open
- EAR ≈ 0.0 when eyes are closed

**Key Concepts:**
- Mediapipe Face Mesh (468 landmarks)
- Euclidean distance for EAR calculation
- Real-time EAR graph visualization
- `winsound.Beep()` for audible alarm
- Blink counter

**Run:**
```bash
python task_09_drowsiness_monitor/drowsiness_monitor.py
python task_09_drowsiness_monitor/drowsiness_monitor.py --ear-threshold 0.20
```

**Controls:** `+/-` adjust threshold | `r` reset | `q/ESC` quit

---

### Task 10: AR Filter Overlay

**File:** `task_10_ar_filter_overlay/ar_filter.py`

**Objective:** Detect facial features and render 2D visual overlays mapped to face landmarks.

**Available Filters:**

| Key | Filter |
|-----|--------|
| 0 | No Filter |
| 1 | Sunglasses |
| 2 | Devil Horns |
| 3 | Mustache |
| 4 | Clown Nose |
| 5 | Full Combo (all) |

**Key Concepts:**
- Mediapipe Face Mesh landmark localization
- Alpha channel blending (`cv2.addWeighted()`)
- Affine transformations for overlay alignment
- Multi-face support

**Run:**
```bash
python task_10_ar_filter_overlay/ar_filter.py
```

**Controls:** `0-5` switch filter | `s` save snapshot | `q/ESC` quit

---

## Key Concepts Summary

| Concept | Tasks | OpenCV Function |
|---------|-------|-----------------|
| Frame Differencing | 1 | `cv2.absdiff()` |
| Contour Detection | 1, 5, 6 | `cv2.findContours()` |
| Color Thresholding (HSV) | 6, 7 | `cv2.inRange()` |
| Edge Detection | 5 | `cv2.Canny()` |
| Perspective Transform | 5 | `cv2.getPerspectiveTransform()` |
| Face Detection | 4, 9, 10 | Mediapipe Face Mesh |
| Hand Tracking | 3 | Mediapipe Hands |
| Object Tracking | 7 | Moment-based centroid |
| OCR | 8 | Pytesseract |
| Image Overlay | 10 | `cv2.addWeighted()` |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Cannot connect to IP Webcam | Ensure phone & PC are on the same Wi-Fi. Check firewall settings. |
| `cv2.imshow()` window not responding | Press any key. Ensure `cv2.waitKey()` is called in the loop. |
| `face_recognition` install fails | Install Visual Studio Build Tools first, then `pip install dlib face_recognition`. |
| Tesseract not found | Download from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH. |
| Mediapipe import error | `pip install mediapipe` — requires Python 3.8-3.11. |
| `pyzbar` not working | Install [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe). |
| Low FPS on IP stream | Reduce resolution in IP Webcam app settings. Use `--source 0` for local webcam. |

---

## License

This project is created for educational purposes as part of a Computer Vision Lab course. Feel free to use and modify for learning.

---

**Author:** Atul Krishna  
**Course:** Computer Vision Lab — Lab Sheet 11  
**Date:** October 2026
