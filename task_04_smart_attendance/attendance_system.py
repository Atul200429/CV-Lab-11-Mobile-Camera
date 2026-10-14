"""
Task 4: Smart Attendance System via Face Recognition
=====================================================
Detects faces in the camera stream, matches them against a local
directory of known faces, and marks attendance automatically in a
CSV file.

Setup:
    1. Place photos of known people in the 'known_faces/' directory.
       - Name files as: PersonName.jpg  (e.g., John_Doe.jpg)
       - Multiple photos: John_Doe_1.jpg, John_Doe_2.jpg
    2. Run the script.

Key Skills: face_recognition library, facial feature embeddings, CSV.

Controls:
    q / ESC  →  Quit
    r        →  Reload known faces
    a        →  Show today's attendance

Usage:
    python attendance_system.py
    python attendance_system.py --source 0
    python attendance_system.py --known-dir ../known_faces
"""

import cv2
import os
import sys
import csv
import argparse
import datetime
import numpy as np

# ── Add project root to path ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.camera_utils import get_camera, release_and_close, put_text

try:
    import face_recognition
except ImportError:
    print("[ERROR] face_recognition library is required.")
    print("        Install: pip install face_recognition")
    print("        Also requires dlib. See: https://github.com/ageitgey/face_recognition")
    sys.exit(1)

# ── Configuration ──
KNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), "..", "known_faces")
ATTENDANCE_DIR = os.path.join(os.path.dirname(__file__), "..", "inventory_log")
TOLERANCE = 0.5           # Lower = stricter matching (0.4-0.6 typical)
FRAME_SCALE = 0.25         # Process at 1/4 resolution for speed
PROCESS_EVERY_N = 3        # Process every Nth frame for performance


def parse_args():
    parser = argparse.ArgumentParser(description="Smart Attendance System")
    parser.add_argument("--source", default=None)
    parser.add_argument("--known-dir", default=KNOWN_FACES_DIR,
                        help="Directory containing known face images")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE,
                        help="Face matching tolerance (lower = stricter)")
    return parser.parse_args()


def load_known_faces(directory):
    """Load and encode all face images from the known_faces directory."""
    known_encodings = []
    known_names = []

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"[WARN] No known faces directory found. Created: {directory}")
        print("       Add face images (PersonName.jpg) and press 'r' to reload.")
        return known_encodings, known_names

    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = [f for f in os.listdir(directory)
             if f.lower().endswith(valid_ext)]

    if not files:
        print(f"[WARN] No face images found in: {directory}")
        print("       Add images named as PersonName.jpg and press 'r' to reload.")
        return known_encodings, known_names

    print(f"[INFO] Loading {len(files)} face image(s)...")

    for filename in files:
        filepath = os.path.join(directory, filename)
        # Extract name: remove extension and trailing numbers (John_Doe_1.jpg → John Doe)
        name = os.path.splitext(filename)[0]
        # Remove trailing _N pattern
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            name = parts[0]
        name = name.replace("_", " ")

        try:
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)
                print(f"  ✔ Loaded: {name} ({filename})")
            else:
                print(f"  ✘ No face found in: {filename}")
        except Exception as e:
            print(f"  ✘ Error loading {filename}: {e}")

    print(f"[INFO] {len(known_encodings)} face(s) loaded successfully.")
    return known_encodings, known_names


def get_attendance_file():
    """Get today's attendance CSV file path."""
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    date_str = datetime.date.today().strftime("%Y-%m-%d")
    filepath = os.path.join(ATTENDANCE_DIR, f"attendance_{date_str}.csv")

    if not os.path.exists(filepath):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Time", "Status"])

    return filepath


def mark_attendance(filepath, name, marked_set):
    """Mark a person as present if not already marked today."""
    if name in marked_set:
        return False

    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([name, timestamp, "Present"])

    marked_set.add(name)
    return True


def load_already_marked(filepath):
    """Load names already marked today."""
    marked = set()
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row:
                    marked.add(row[0])
    return marked


def main():
    args = parse_args()

    source = args.source
    if source is not None:
        try:
            source = int(source)
        except ValueError:
            pass

    # Load known faces
    known_encodings, known_names = load_known_faces(args.known_dir)

    # Attendance setup
    attendance_file = get_attendance_file()
    marked_today = load_already_marked(attendance_file)

    cap = get_camera(source=source)
    frame_count = 0
    face_locations_display = []
    face_names_display = []

    print("=" * 60)
    print("  SMART ATTENDANCE SYSTEM")
    print("=" * 60)
    print(f"  Known faces : {len(known_encodings)}")
    print(f"  Tolerance   : {args.tolerance}")
    print(f"  Attendance  : {os.path.abspath(attendance_file)}")
    print(f"  Marked today: {len(marked_today)}")
    print("  Controls    : q/ESC quit | r reload faces | a show attendance")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1

        # Process every Nth frame for performance
        if frame_count % PROCESS_EVERY_N == 0 and known_encodings:
            # Resize for faster processing
            small_frame = cv2.resize(frame, (0, 0),
                                     fx=FRAME_SCALE, fy=FRAME_SCALE)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            face_locations = face_recognition.face_locations(rgb_small)
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            face_locations_display = []
            face_names_display = []

            for encoding, location in zip(face_encodings, face_locations):
                # Compare with known faces
                distances = face_recognition.face_distance(known_encodings, encoding)
                matches = face_recognition.compare_faces(
                    known_encodings, encoding, tolerance=args.tolerance
                )

                name = "Unknown"
                confidence = 0.0

                if True in matches:
                    best_idx = np.argmin(distances)
                    if matches[best_idx]:
                        name = known_names[best_idx]
                        confidence = 1.0 - distances[best_idx]

                        # Mark attendance
                        if mark_attendance(attendance_file, name, marked_today):
                            print(f"  ✔ ATTENDANCE: {name} marked present!")

                # Scale back location to full frame
                top, right, bottom, left = location
                scale = int(1 / FRAME_SCALE)
                face_locations_display.append(
                    (top * scale, right * scale, bottom * scale, left * scale)
                )
                face_names_display.append((name, confidence))

        # Draw results
        for (top, right, bottom, left), (name, conf) in zip(
            face_locations_display, face_names_display
        ):
            if name == "Unknown":
                color = (0, 0, 255)  # Red for unknown
            else:
                color = (0, 255, 0)  # Green for recognized

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Name label
            label = f"{name}"
            if conf > 0:
                label += f" ({conf:.0%})"
            cv2.rectangle(frame, (left, bottom), (right, bottom + 25), color, -1)
            cv2.putText(frame, label, (left + 5, bottom + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

        # Header
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (30, 30, 30), -1)
        put_text(frame, f"Attendance | Present: {len(marked_today)} | "
                        f"Known: {len(known_encodings)}",
                 (10, 28), color=(0, 255, 0), scale=0.6)

        cv2.imshow("Smart Attendance System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            known_encodings, known_names = load_known_faces(args.known_dir)
            print(f"[INFO] Reloaded {len(known_encodings)} face(s).")
        elif key == ord('a'):
            print(f"\n  Today's attendance ({len(marked_today)}):")
            for name in sorted(marked_today):
                print(f"    ✔ {name}")
            print()

    print(f"\n[INFO] Attendance system stopped.")
    print(f"[INFO] Today's log: {os.path.abspath(attendance_file)}")
    release_and_close(cap)


if __name__ == "__main__":
    main()
