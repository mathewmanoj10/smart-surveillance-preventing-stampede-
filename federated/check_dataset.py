
from pathlib import Path
import random
import cv2

DATASET = Path("crowd_yolo_dataset")
IMAGE_DIR = DATASET / "images" / "train"
LABEL_DIR = DATASET / "labels" / "train"

images = list(IMAGE_DIR.glob("*"))
random.seed(42)
random.shuffle(images)

shown = 0

for image_path in images:
    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read:", image_path.name)
        continue

    label_path = LABEL_DIR / (image_path.stem + ".txt")

    if not label_path.exists():
        print("Missing label:", image_path.name)
        continue

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id, xc, yc, bw, bh = map(float, parts)

        h, w = image.shape[:2]
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        cv2.rectangle(
            image, (x1, y1), (x2, y2), (0, 255, 0), 2
        )
        cv2.putText(
            image, "person", (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

    cv2.imshow("Check person annotations", image)
    print(image_path.name, "| Person boxes:", len(lines))

    key = cv2.waitKey(0)

    if key == ord("q"):
        break

    shown += 1
    if shown >= 5:
        break

cv2.destroyAllWindows()
