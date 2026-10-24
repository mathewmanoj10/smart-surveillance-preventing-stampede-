import os
import cv2
import torch

from local_clients.client import LocalClient
from federated.fed_avg import (
    federated_average,
    aggregate_risk,
    print_risk_report,
)
from models.simple_model import SimpleDensityModel
from models.csrnet import load_csrnet
from utils.scene_profiles import SCENE_PROFILES


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_DIR = os.path.join("federated", "videos")

VIDEO_PATHS = [
    os.path.join(
        VIDEO_DIR,
        "stampede",
        "gettyimages-2234292699-640_adpp.mp4",
    )
]

CLIENT_NAMES = ["Camera 1"]

# Scene type for each camera. Options (see utils/scene_profiles.py):
#   "public_street", "road_traffic", "railway_platform",
#   "event_venue", "religious_gathering"
CLIENT_SCENES = ["public_street"]

# Restricted zone for each camera (roadway, platform edge, entrance...).
# None = no zone. Otherwise a list of normalized (x, y) points, 0..1.
# Get them by running:  python pick_zone.py "<video path>"
# Example: [(0.10, 0.60), (0.90, 0.60), (0.95, 0.95), (0.05, 0.95)]
CLIENT_ZONES = [None]

# Vehicle detector, only loaded if a camera uses a scene that tracks vehicles
VEHICLE_WEIGHTS = "yolo11n.pt"
SHOW_VEHICLE_BOXES = False   # boxes update with CSRNet, so they can look laggy

# SIMULATED regulation controller (signs, announcements, gate holds...).
# Nothing here controls real hardware; actions are shown on screen and
# written to outputs/<camera>_actions.csv.
REGULATE = True
APPROVAL = "operator"        # "operator" = press A/X in the video window
                             # "auto"     = actions activate without prompts

# Optional training labels (training only): 0.0 = normal, 1.0 = stampede.
# Use None to train on the formula risk instead.
CLIENT_LABELS = [None]

MAX_FRAMES = None        # None = whole video
MAX_SIDE = 512           # CSRNet input size (smaller = faster)
COUNT_EVERY = 8          # run CSRNet once every N frames
PX_PER_METER = None      # CALIBRATE: pixels per metre in the crowd area

DISPLAY = True
SHOW_HEATMAP = False

CSRNET_WEIGHTS = os.path.join("weights", "PartAmodel_best.pth.tar")
LWCC_WEIGHTS = "SHA"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOAD CROWD-COUNTING MODEL
# ============================================================

def load_counting_model():

    if os.path.isfile(CSRNET_WEIGHTS):
        try:
            model = load_csrnet(CSRNET_WEIGHTS, device=DEVICE)

            if hasattr(model, "to"):
                model = model.to(DEVICE)

            if hasattr(model, "eval"):
                model.eval()

            print("CSRNet loaded from:", CSRNET_WEIGHTS)
            return model, "csrnet"

        except Exception as error:
            print("Local CSRNet loading failed:", error)

    try:
        from lwcc import LWCC

        model = LWCC.load_model(
            model_name="CSRNet",
            model_weights=LWCC_WEIGHTS
        )

        print(f"LWCC CSRNet backend loaded ({LWCC_WEIGHTS}).")
        return model, "lwcc"

    except Exception as error:
        raise RuntimeError(
            "Could not load CSRNet or LWCC. "
            "Check your model weights and installation."
        ) from error


# ============================================================
# LOAD VEHICLE MODEL (only for road-type scenes)
# ============================================================

def load_vehicle_model(scenes):
    """Loads YOLO11 only if at least one camera needs vehicle tracking."""

    needs_vehicles = any(
        SCENE_PROFILES[s].get("track_vehicles") for s in scenes
    )

    if not needs_vehicles:
        return None

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[WARNING] Vehicle tracking needs: pip install ultralytics")
        return None

    path = os.path.join("federated", VEHICLE_WEIGHTS)

    if not os.path.isfile(path):
        path = VEHICLE_WEIGHTS   # ultralytics downloads it if internet is on

    try:
        model = YOLO(path)
        print("Vehicle model loaded:", path)
        return model
    except Exception as error:
        print("[WARNING] Could not load vehicle model:", error)
        return None


# ============================================================
# FEDERATED PIPELINE
# ============================================================

def run_federated_pipeline():

    print("=" * 60)
    print("SMART SURVEILLANCE: PREVENTING URBAN STAMPEDES")
    print("Device:", DEVICE)
    print("=" * 60)

    # Validate scene names early so a typo fails with a clear message
    for scene in CLIENT_SCENES:
        if scene not in SCENE_PROFILES:
            print(f"[ERROR] Unknown scene '{scene}'. "
                  f"Choose from: {', '.join(SCENE_PROFILES)}")
            return

    try:
        count_model, backend_type = load_counting_model()
    except Exception as error:
        print("[ERROR]", error)
        return

    vehicle_model = load_vehicle_model(CLIENT_SCENES)

    clients = []
    labels = []

    for index, video_path in enumerate(VIDEO_PATHS):

        if not os.path.isfile(video_path):
            print("[WARNING] Video not found:", os.path.abspath(video_path))
            continue

        name = (
            CLIENT_NAMES[index]
            if index < len(CLIENT_NAMES)
            else f"Camera {index + 1}"
        )

        scene = (
            CLIENT_SCENES[index]
            if index < len(CLIENT_SCENES)
            else "public_street"
        )

        zone = CLIENT_ZONES[index] if index < len(CLIENT_ZONES) else None
        label = CLIENT_LABELS[index] if index < len(CLIENT_LABELS) else None

        clients.append(
            LocalClient(
                client_id=name,
                video_path=video_path,
                count_model=count_model,
                backend=backend_type,
                device=DEVICE,
                display=DISPLAY,
                max_side=MAX_SIDE,
                show_heatmap=SHOW_HEATMAP,
                count_every=COUNT_EVERY,
                px_per_meter=PX_PER_METER,
                scene=scene,
                zone_polygon=zone,
                vehicle_model=vehicle_model,
                show_vehicle_boxes=SHOW_VEHICLE_BOXES,
                regulate=REGULATE,
                approval=APPROVAL,
            )
        )
        labels.append(label)

    if not clients:
        print("No valid videos found. Check VIDEO_PATHS in main.py.")
        return

    # --------------------------------------------------------
    # Process each camera and train its local model
    # --------------------------------------------------------

    for client, label in zip(clients, labels):

        print(f"\nProcessing: {client.client_id} "
              f"(scene: {client.scene.label})")

        try:
            summary = client.process_video(max_frames=MAX_FRAMES)

            if summary is None:
                print(f"[WARNING] Could not process {client.client_id}.")
                continue

            client.simulate_local_training(epochs=30, target_label=label)

        except Exception as error:
            print(f"[ERROR] {client.client_id} failed:", error)
            raise

    trained = [c for c in clients if c.frames_processed > 0]

    if not trained:
        print("No clients successfully processed any frames.")
        return

    # --------------------------------------------------------
    # 1) Federated averaging of the small model
    # --------------------------------------------------------

    print("\nAggregating local models (FedAvg)...")

    global_state = federated_average(
        [c.get_model_weights() for c in trained],
        [c.frames_processed for c in trained],
    )

    global_model = SimpleDensityModel(input_size=3)
    global_model.load_state_dict(global_state)

    save_path = os.path.join(OUTPUT_DIR, "global_model.pth")
    torch.save(global_model.state_dict(), save_path)
    print("Global model saved:", os.path.abspath(save_path))

    # --------------------------------------------------------
    # 2) Risk report across cameras
    # --------------------------------------------------------

    updates = [c.get_risk_update() for c in trained]
    print_risk_report(updates, aggregate_risk(updates))

    print("\nFINAL CAMERA SUMMARY")
    print("=" * 60)

    for client in trained:
        client.print_summary()
        print()

    print("FEDERATED LEARNING PIPELINE COMPLETED")


# ============================================================
# OPTIONAL YOLO PREVIEW
# ============================================================

def run_fast_video_preview(video_path):

    if not os.path.isfile(video_path):
        print("Video not found:", os.path.abspath(video_path))
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        print("YOLO preview requires: pip install ultralytics")
        return

    yolo_path = os.path.join("federated", "yolo11n.pt")

    if not os.path.isfile(yolo_path):
        yolo_path = "yolo11n.pt"

    if not os.path.isfile(yolo_path):
        print("YOLO weights not found:", yolo_path)
        return

    model = YOLO(yolo_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open video:", video_path)
        return

    processed = 0
    limit = MAX_FRAMES if MAX_FRAMES is not None else float("inf")

    try:
        while processed < limit:

            ret, frame = cap.read()

            if not ret:
                break

            h, w = frame.shape[:2]
            scale = min(1.0, MAX_SIDE / max(h, w))

            if scale < 1.0:
                frame = cv2.resize(
                    frame,
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA
                )

            result = model.predict(
                source=frame, classes=[0], conf=0.25,
                verbose=False, device=str(DEVICE)
            )[0]

            count = len(result.boxes) if result.boxes is not None else 0
            annotated = result.plot()

            cv2.putText(
                annotated, f"People detected: {count}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

            cv2.imshow("Fast YOLO Video Detection", annotated)
            processed += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("YOLO preview finished. Frames processed:", processed)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\nChoose a mode:")
    print("1. Fast YOLO video detection")
    print("2. CSRNet/LWCC + risk analysis + federated learning")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        run_fast_video_preview(VIDEO_PATHS[0])
    elif choice == "2":
        run_federated_pipeline()
    else:
        print("Invalid choice.")