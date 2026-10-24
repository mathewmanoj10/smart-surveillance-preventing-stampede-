
import json
import urllib.request
from pathlib import Path
from urllib.parse import urlparse
import yaml

# CHANGE THIS to your NDJSON file's actual path
NDJSON_PATH = Path(r"C:\Users\user\OneDrive\Desktop\Capstone project\federated\example-dataset (1).ndjson")

OUTPUT_DIR = Path("crowd_yolo_dataset")

# Create YOLO dataset folders
for split in ["train", "val"]:
    (OUTPUT_DIR / "images" / split).mkdir(
        parents=True, exist_ok=True
    )
    (OUTPUT_DIR / "labels" / split).mkdir(
        parents=True, exist_ok=True
    )

records = []

with open(NDJSON_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

metadata = next(
    (r for r in records if r.get("type") == "dataset"),
    {}
)

print("Dataset name:", metadata.get("name"))
print("Image records:", sum(
    r.get("type") == "image" for r in records
))

success = 0
failed = []

for record in records:
    if record.get("type") != "image":
        continue

    split = record.get("split")

    if split not in ["train", "val"]:
        print("Skipping unknown split:", split)
        continue

    filename = Path(record["file"]).name
    image_path = OUTPUT_DIR / "images" / split / filename
    label_path = OUTPUT_DIR / "labels" / split / (
        Path(filename).stem + ".txt"
    )

    try:
        # Download the image if it isn't already present
        if not image_path.exists():
            image_url = record.get("url")

            if not image_url:
                raise ValueError("Image URL is missing")

            request = urllib.request.Request(
                image_url,
                headers={"User-Agent": "Mozilla/5.0"}
            )

            with urllib.request.urlopen(request, timeout=30) as response:
                image_path.write_bytes(response.read())

        # Keep only person boxes (class ID 0)
        boxes = record.get("annotations", {}).get("boxes", [])
        yolo_lines = []

        for box in boxes:
            if len(box) != 5:
                continue

            class_id, x, y, w, h = box

            if int(class_id) != 0:
                continue

            yolo_lines.append(
                f"0 {float(x):.6f} {float(y):.6f} "
                f"{float(w):.6f} {float(h):.6f}"
            )

        label_path.write_text(
            "\n".join(yolo_lines),
            encoding="utf-8"
        )

        success += 1
        print(f"Processed {split}: {filename}")

    except Exception as e:
        failed.append((filename, str(e)))
        print(f"FAILED: {filename} - {e}")

# Create the YOLO configuration file
yaml_content = {
    "path": str(OUTPUT_DIR.resolve()),
    "train": "images/train",
    "val": "images/val",
    "names": {0: "person"}
}

with open(OUTPUT_DIR / "data.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(yaml_content, f, sort_keys=False)

print("\n========== SUMMARY ==========")
print("Images processed:", success)
print("Images failed:", len(failed))
print("Dataset folder:", OUTPUT_DIR.resolve())
print("YOLO config:", OUTPUT_DIR / "data.yaml")

if failed:
    print("\nFirst failures:")
    for filename, error in failed[:10]:
        print(filename, "->", error)
