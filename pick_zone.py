"""
pick_zone.py

Click a zone (roadway, platform edge, entrance...) on a video frame and
print its coordinates, ready to paste into CLIENT_ZONES in main.py.

Usage (from the project root):
    python pick_zone.py "path/to/video.mp4"
    python pick_zone.py "path/to/video.mp4" 120      # use frame number 120

Controls:
    left click : add a point
    z          : undo last point
    r          : reset
    Enter      : finish and print the coordinates
    q / Esc    : quit without printing
"""

import sys

import cv2
import numpy as np

MAX_W = 1280
WINDOW = "Pick zone"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    video_path = sys.argv[1]
    frame_no = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open video:", video_path)
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        print("Could not read that frame.")
        return

    h, w = frame.shape[:2]
    scale = min(1.0, MAX_W / w)

    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    sh, sw = frame.shape[:2]
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        canvas = frame.copy()

        if points:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(canvas, [pts], len(points) > 2, (0, 255, 255), 2)

            for p in points:
                cv2.circle(canvas, p, 4, (0, 0, 255), -1)

        cv2.putText(
            canvas, "Click points | z undo | r reset | Enter done | q quit",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        cv2.imshow(WINDOW, canvas)

        key = cv2.waitKey(30) & 0xFF

        if key in (27, ord("q")):
            print("Cancelled.")
            break

        if key == ord("z") and points:
            points.pop()

        if key == ord("r"):
            points.clear()

        if key in (13, 10):
            if len(points) < 3:
                print("Need at least 3 points.")
                continue

            normalized = [(round(x / sw, 3), round(y / sh, 3))
                          for x, y in points]

            print("\nPaste this into CLIENT_ZONES in main.py:\n")
            print(normalized)
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()