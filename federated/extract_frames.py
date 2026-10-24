
import cv2
import os

VIDEO_FOLDER = "videos"
OUTPUT_FOLDER = "frames"

# Extract one frame every 2 seconds
SECONDS_INTERVAL = 2

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

total = 0

for filename in sorted(os.listdir(VIDEO_FOLDER)):

    if not filename.lower().endswith(
        (".mp4", ".avi", ".mov", ".mkv")
    ):
        continue

    path = os.path.join(VIDEO_FOLDER, filename)
    cap = cv2.VideoCapture(path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        print("Could not read:", filename)
        cap.release()
        continue

    interval = max(1, int(fps * SECONDS_INTERVAL))
    video_name = os.path.splitext(filename)[0]
    frame_number = 0
    saved = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_number % interval == 0:
            image_name = (
                f"{video_name}_{frame_number:07d}.jpg"
            )

            cv2.imwrite(
                os.path.join(OUTPUT_FOLDER, image_name),
                frame
            )

            saved += 1
            total += 1

        frame_number += 1

    cap.release()
    print(filename, ":", saved, "images extracted")

print("Total images:", total)
