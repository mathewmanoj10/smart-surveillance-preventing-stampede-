import os
import csv
import time
import tempfile
import threading
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.density_utils import calculate_density, FPSCounter
from models.simple_model import SimpleDensityModel
from utils.scene_profiles import SceneAnalyzer
from utils.regulation import RegulationController


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Motion / risk constants
MOVING_THRESHOLD = 0.3     # m/s, slower pixels are treated as standing
MIN_MOVER_PEOPLE = 2.0     # people-equivalents needed to trust direction variance
W_DENSITY, W_SPEED, W_DIR, W_INTERACT = 0.35, 0.25, 0.20, 0.20
EMA_ALPHA = 0.30
LOW_T, HIGH_T = 0.35, 0.65

# Smoothing for the raw optical-flow readings (reduces frame-to-frame jitter)
MOTION_EMA = 0.40

# Heatmap display
HEAT_EMA = 0.25            # how fast the heatmap fades toward a new CSRNet result
HEAT_MIN = 0.12            # fraction of peak density below which nothing is tinted

# Heatmap colour scheme. Try any of:
#   cv2.COLORMAP_JET      blue -> green -> yellow -> red   (classic crowd look)
#   cv2.COLORMAP_TURBO    smoother version of JET
#   cv2.COLORMAP_HOT      black -> red -> yellow -> white  (no blue)
#   cv2.COLORMAP_INFERNO  black -> purple -> orange -> yellow
#   cv2.COLORMAP_OCEAN    dark -> blue/teal -> white       (blue-toned)
HEAT_COLORMAP = cv2.COLORMAP_JET


def clip01(x):
    return float(min(max(x, 0.0), 1.0))


class LocalClient:

    def __init__(
        self,
        client_id,
        video_path,
        count_model,
        device="cpu",
        display=True,
        max_side=512,            # smaller = faster CSRNet
        show_heatmap=False,
        heatmap_alpha=0.45,
        backend="csrnet",
        count_every=8,           # submit a CSRNet job at most once every N frames
        flow_width=320,          # optical flow resolution
        px_per_meter=None,       # CALIBRATE THIS (pixels per metre in the crowd area)
        window_m=2.0,            # density window side in metres
        density_critical=6.0,    # people/m2 that maps to density score 1.0
        speed_critical=2.5,      # m/s that maps to speed score 1.0 (running)
        write_video=True,
        csv_dir="outputs",
        async_inference=True,    # run CSRNet in a background thread (display mode)
        pace_to_video=True,      # play back at the video's own FPS
        display_width=1280,      # downscale frames wider than this for display only
        scene="public_street",   # key of SCENE_PROFILES (utils/scene_profiles.py)
        zone_polygon=None,       # [(x, y), ...] normalized 0..1 (roadway, platform edge...)
        vehicle_model=None,      # loaded YOLO11 model (road-type scenes only)
        show_vehicle_boxes=False,
        regulate=True,           # run the SIMULATED regulation controller
        approval="operator",     # "operator" = press A/X, "auto" = no prompts
    ):
        self.client_id = client_id
        self.video_path = video_path
        self.count_model = count_model
        self.device = torch.device(device)
        self.display = display
        self.max_side = max_side
        self.show_heatmap = show_heatmap
        self.heatmap_alpha = heatmap_alpha
        self.backend = backend.lower()
        self.count_every = max(1, int(count_every))
        self.flow_width = int(flow_width)
        self.px_per_meter = px_per_meter
        self.window_m = window_m
        self.density_critical = density_critical
        self.speed_critical = speed_critical
        self.write_video = write_video
        self.csv_dir = csv_dir
        self.pace_to_video = pace_to_video
        self.display_width = display_width

        # Scene profile: sets this camera's thresholds and decides which
        # extra measurements (vehicles, restricted zone) are made.
        self.scene = SceneAnalyzer(
            scene,
            zone_polygon=zone_polygon,
            vehicle_model=vehicle_model,
            draw_vehicles=show_vehicle_boxes,
        )
        self.density_critical = self.scene.density_critical
        self.speed_critical = self.scene.speed_critical
        self.low_t = self.scene.low_t
        self.high_t = self.scene.high_t

        # Simulated regulation: turns the alert level into control actions.
        # Without a display there is nobody to press A/X, so it auto-approves.
        self.controller = None

        if regulate:
            tag = self.client_id.replace(" ", "_")
            self.controller = RegulationController(
                scene,
                log_path=os.path.join(csv_dir, f"{tag}_actions.csv"),
                title=f"Control panel (simulated) - {self.client_id}",
                approval=approval if display else "auto",
            )

        # Background inference only makes sense for live playback. For offline
        # runs (display=False) the loop runs as fast as possible, so we keep
        # CSRNet synchronous to make sure every density map matches its frame.
        self.async_inference = bool(async_inference and display)

        if self.backend not in ("csrnet", "lwcc"):
            raise ValueError("backend must be 'csrnet' or 'lwcc'")

        self._tmp_path = os.path.join(
            tempfile.gettempdir(), f"lwcc_frame_{id(self)}.jpg"
        )

        # Small model for the federated-learning demo.
        # Inputs: [density score, speed score, direction score]
        self.local_model = SimpleDensityModel(
            input_size=3, hidden_size=8, output_size=1
        )

        self.use_half = (
            self.backend == "csrnet" and self.device.type == "cuda"
        )

        if self.backend == "csrnet" and hasattr(self.count_model, "eval"):
            self.count_model.eval()
            if self.use_half:
                self.count_model.half()

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.local_stats = {
            "person_counts": [],
            "densities": [],        # peak people / m2
            "speeds": [],           # m/s
            "direction_variances": [],
            "risk_scores": [],      # 0..100
            "features": [],         # [D, S, V] in 0..1
            "risk_unit": [],        # 0..1
            "fps_values": [],
        }

        self.fps_counter = FPSCounter()
        self.fps_window = deque(maxlen=30)
        self.count_buffer = deque(maxlen=3)
        self.frames_processed = 0

        self._dm = None
        self._in_size = None
        self._peak_density = 0.0
        self._flow_weights = None
        self._risk_smooth = 0.0
        self._speed_s = 0.0
        self._dir_s = 0.0

        # Heatmap state (kept at low resolution, upscaled only for drawing)
        self._dm_version = 0
        self._heat_target_version = -1
        self._heat_target = None
        self._heat_cur = None
        self._heat_peak = 1e-6

        # Background inference state
        self._lock = threading.Lock()
        self._frame_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker = None
        self._pending = None
        self._result = None
        self._busy = False

    # ------------------------------------------------------
    # CROWD COUNTING (CSRNet / LWCC)
    # ------------------------------------------------------

    def _estimate_crowd(self, frame):
        h, w = frame.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))

        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )

        in_h, in_w = frame.shape[:2]

        if self.backend == "lwcc":
            count, dm = self._estimate_with_lwcc(frame)
            return count, dm, (in_w, in_h)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD

        tensor = torch.from_numpy(
            rgb.transpose(2, 0, 1).copy()
        ).unsqueeze(0).to(self.device)

        if self.use_half:
            tensor = tensor.half()

        with torch.inference_mode():
            output = self.count_model(tensor)

        dm = output.squeeze().float().cpu().numpy()

        if dm.ndim != 2:
            raise RuntimeError(
                f"Expected a 2D CSRNet density map, got {dm.shape}"
            )

        return float(max(dm.sum(), 0.0)), dm, (in_w, in_h)

    def _estimate_with_lwcc(self, frame):
        from lwcc import LWCC

        if not cv2.imwrite(self._tmp_path, frame):
            raise RuntimeError("Could not save temporary video frame.")

        result = LWCC.get_count(
            self._tmp_path, model=self.count_model, return_density=True
        )

        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise RuntimeError("LWCC did not return (count, density_map).")

        count, dm = result
        dm = np.squeeze(np.asarray(dm))

        if dm.ndim != 2:
            raise RuntimeError(f"Expected a 2D LWCC map, got {dm.shape}")

        return float(max(float(count), 0.0)), dm

    def _run_jobs(self, frame):
        """Everything heavy for one frame: crowd density + (optional) vehicles."""
        raw_count, dm, in_size = self._estimate_crowd(frame)
        vehicles = self.scene.detect_vehicles(frame)
        return raw_count, dm, in_size, vehicles

    def _apply_estimate(self, raw_count, dm, in_size, vehicles=None):
        """Store a fresh CSRNet result (always called from the main thread)."""
        self.count_buffer.append(raw_count)
        self._dm_version += 1
        self._dm = dm
        self._in_size = in_size
        self._peak_density = self._peak_density_from_map(dm)
        self._update_flow_weights(dm)

        self.scene.update(
            dm,
            float(np.mean(self.count_buffer)),
            vehicles,
            (self.orig_w, self.orig_h),
            (self.flow_w, self.flow_h),
        )

    # ------------------------------------------------------
    # BACKGROUND INFERENCE THREAD
    # ------------------------------------------------------

    def _inference_worker(self):
        """Waits for a frame, runs CSRNet on it, publishes the result."""
        while not self._stop_event.is_set():
            if not self._frame_event.wait(timeout=0.1):
                continue

            self._frame_event.clear()

            with self._lock:
                frame = self._pending
                self._pending = None

            if frame is None:
                continue

            try:
                result = self._run_jobs(frame)
            except Exception as error:   # re-raised in the main thread
                result = error

            with self._lock:
                self._result = result
                self._busy = False

    def _start_worker(self):
        self._stop_event.clear()
        self._frame_event.clear()
        self._pending = None
        self._result = None
        self._busy = False
        self._worker = threading.Thread(
            target=self._inference_worker, daemon=True
        )
        self._worker.start()

    def _stop_worker(self):
        if self._worker is not None:
            self._stop_event.set()
            self._frame_event.set()
            self._worker.join(timeout=5)
            self._worker = None

    def _submit(self, frame):
        with self._lock:
            self._pending = frame.copy()
            self._busy = True
        self._frame_event.set()

    def _collect_result(self):
        with self._lock:
            result, self._result = self._result, None

        if result is None:
            return

        if isinstance(result, Exception):
            raise result

        self._apply_estimate(*result)

    def _worker_busy(self):
        with self._lock:
            return self._busy

    # ------------------------------------------------------
    # DENSITY IN PEOPLE / M2
    # ------------------------------------------------------

    def _peak_density_from_map(self, dm):
        """Peak local density in people/m2 using a window_m x window_m window."""
        dm = np.clip(dm, 0, None).astype(np.float32)
        map_h, map_w = dm.shape

        ppm_map = self.px_per_meter * map_w / self.orig_w
        k = int(round(self.window_m * ppm_map))
        k = max(3, min(k, map_h, map_w))

        sums = cv2.boxFilter(
            dm, -1, (k, k), normalize=False,
            borderType=cv2.BORDER_CONSTANT,
        )

        area_m2 = (k / ppm_map) ** 2
        return float(np.percentile(sums, 99) / area_m2)

    def _update_flow_weights(self, dm):
        """Resize the density map to the flow grid, preserving its total."""
        dm = np.clip(dm, 0, None).astype(np.float32)
        w = cv2.resize(
            dm, (self.flow_w, self.flow_h), interpolation=cv2.INTER_LINEAR
        )
        s = float(w.sum())
        if s > 1e-6:
            w *= float(dm.sum()) / s
        self._flow_weights = w.astype(np.float32)

    # ------------------------------------------------------
    # MOTION: SPEED + DIRECTION VARIANCE
    # ------------------------------------------------------

    def _flow_stats(self, prev_gray, gray):
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 2, 15, 2, 5, 1.1, 0
        )
        fx, fy = flow[..., 0], flow[..., 1]
        mag_px = np.hypot(fx, fy)
        speed_ms = mag_px * self.flow_m_per_px / self.dt

        w = self._flow_weights
        total_w = float(w.sum()) if w is not None else 0.0

        if total_w < 1e-6:
            return 0.0, 0.0

        mean_speed = float((w * speed_ms).sum() / total_w)

        wm = w * (speed_ms > MOVING_THRESHOLD)
        mover_w = float(wm.sum())

        if mover_w < MIN_MOVER_PEOPLE:
            return mean_speed, 0.0

        safe = np.maximum(mag_px, 1e-6)
        ux, uy = fx / safe, fy / safe
        R = np.hypot((wm * ux).sum(), (wm * uy).sum()) / mover_w

        return mean_speed, clip01(1.0 - float(R))

    # ------------------------------------------------------
    # RISK
    # ------------------------------------------------------

    def _risk(self, density, speed, dir_var):
        D = clip01(density / self.density_critical)
        S = clip01(speed / self.speed_critical)
        V = clip01(dir_var)

        raw = (
            W_DENSITY * D
            + W_SPEED * S
            + W_DIR * V
            + W_INTERACT * D * V
        )

        self._risk_smooth = (
            EMA_ALPHA * raw + (1 - EMA_ALPHA) * self._risk_smooth
        )

        r = self._risk_smooth

        if r < self.low_t:
            level, color = "LOW", (0, 255, 0)
        elif r < self.high_t:
            level, color = "MEDIUM", (0, 165, 255)
        else:
            level, color = "HIGH", (0, 0, 255)

        return raw, r, level, color, (D, S, V)

    # ------------------------------------------------------
    # HEATMAP
    # ------------------------------------------------------

    def _overlay_heatmap(self, frame, dm):
        """
        Draws the CSRNet density map as a colour heatmap.

        The heavy work happens at the (small) optical-flow resolution:
        the map fades smoothly toward each new CSRNet result instead of
        flashing, and only the final colour image is upscaled to the frame.
        """
        h, w = frame.shape[:2]

        # New CSRNet result -> new fade target
        if self._heat_target_version != self._dm_version:
            target = cv2.resize(
                np.clip(dm, 0, None).astype(np.float32),
                (self.flow_w, self.flow_h),
                interpolation=cv2.INTER_LINEAR,
            )
            peak = max(float(target.max()), 1e-6)

            if self._heat_cur is None:
                self._heat_cur = target.copy()
                self._heat_peak = peak
            else:
                # slow-moving peak keeps colours from flickering
                self._heat_peak = 0.7 * self._heat_peak + 0.3 * peak

            self._heat_target = target
            self._heat_target_version = self._dm_version

        # Fade the displayed map a little closer to the target every frame
        self._heat_cur += HEAT_EMA * (self._heat_target - self._heat_cur)

        norm = np.clip(self._heat_cur / self._heat_peak, 0.0, 1.0)
        norm8 = (norm * 255).astype(np.uint8)

        heat = cv2.applyColorMap(norm8, HEAT_COLORMAP)
        mask = (norm >= HEAT_MIN).astype(np.uint8) * 255

        # Upscale only the final images to the frame size
        heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        blended = cv2.addWeighted(
            frame, 1 - self.heatmap_alpha, heat, self.heatmap_alpha, 0
        )

        return cv2.copyTo(blended, mask, frame)

    # ------------------------------------------------------
    # VIDEO PROCESSING
    # ------------------------------------------------------

    def process_video(self, max_frames=None):
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"[{self.client_id}] Unable to open {self.video_path}")
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_video = cap.get(cv2.CAP_PROP_FPS)

        if not fps_video or fps_video <= 0:
            fps_video = 25.0

        # Geometry for converting pixels to metres
        self.orig_w = width
        self.orig_h = height
        if self.px_per_meter is None:
            # Assumption: the frame spans about 20 m. CALIBRATE for real numbers.
            self.px_per_meter = width / 20.0
            print(
                f"[{self.client_id}] px_per_meter not set; assuming the "
                f"frame is 20 m wide ({self.px_per_meter:.1f} px/m)."
            )

        self.dt = 1.0 / fps_video
        self.flow_w = min(self.flow_width, width)
        self.flow_h = max(1, int(round(height * self.flow_w / width)))
        flow_scale = self.flow_w / width
        self.flow_m_per_px = 1.0 / (self.px_per_meter * flow_scale)

        # Display-only downscale (the saved video keeps full resolution)
        show_size = None
        if self.display and self.display_width and width > self.display_width:
            show_size = (
                self.display_width,
                max(1, int(round(height * self.display_width / width))),
            )

        os.makedirs("outputs", exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)
        tag = self.client_id.replace(" ", "_")

        writer = None
        if self.write_video:
            output_path = os.path.join("outputs", f"{tag}_output.mp4")
            writer = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                fps_video, (width, height),
            )
            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f"Could not create {output_path}")

        csv_path = os.path.join(self.csv_dir, f"{tag}_risk.csv")
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_w = csv.writer(csv_file)
        csv_w.writerow([
            "frame", "time_s", "people", "peak_density_pm2", "speed_ms",
            "direction_variance", "risk_raw", "risk_smooth", "level",
        ] + self.scene.csv_header())

        if self.display:
            cv2.namedWindow(self.client_id, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.client_id, 1280, 720)

        if self.async_inference:
            self._start_worker()

        frame_index = 0
        last_submit = 0
        prev_gray = None
        next_t = time.perf_counter()

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                if max_frames is not None and frame_index >= max_frames:
                    break

                # ---- CSRNet ----
                if self._dm is None:
                    # first frame: run once synchronously so we have a map
                    self._apply_estimate(*self._run_jobs(frame))
                    last_submit = frame_index

                elif self.async_inference:
                    # pick up a finished result (never waits for CSRNet)
                    self._collect_result()

                    if (not self._worker_busy()
                            and frame_index - last_submit >= self.count_every):
                        self._submit(frame)
                        last_submit = frame_index

                elif frame_index % self.count_every == 0:
                    self._apply_estimate(*self._run_jobs(frame))

                smooth_count = int(round(float(np.mean(self.count_buffer))))

                # ---- cheap optical flow every frame ----
                small = cv2.resize(
                    frame, (self.flow_w, self.flow_h),
                    interpolation=cv2.INTER_AREA,
                )
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

                if prev_gray is not None:
                    speed, dir_var = self._flow_stats(prev_gray, gray)
                else:
                    speed, dir_var = 0.0, 0.0

                prev_gray = gray

                # smooth the raw flow readings so numbers don't flicker
                self._speed_s = (
                    MOTION_EMA * speed + (1 - MOTION_EMA) * self._speed_s
                )
                self._dir_s = (
                    MOTION_EMA * dir_var + (1 - MOTION_EMA) * self._dir_s
                )
                speed, dir_var = self._speed_s, self._dir_s

                raw, risk, level, color, (D, S, V) = self._risk(
                    self._peak_density, speed, dir_var
                )

                # Scene rules (restricted zone, traffic) can raise the level
                level, color, alerts = self.scene.apply_rules(level, color)
                self.scene.record()

                # Simulated regulation (uses VIDEO time so results do not
                # depend on how fast this computer processes the video)
                if self.controller is not None:
                    self.controller.update(
                        frame_index / fps_video, frame_index, level,
                        {
                            "people": smooth_count,
                            "density": self._peak_density,
                            "zone": self.scene.zone_people,
                            "risk": risk * 100.0,
                        },
                    )

                fps = self.fps_counter.update()
                self.fps_window.append(fps)
                fps_avg = float(np.mean(self.fps_window))

                st = self.local_stats
                st["person_counts"].append(smooth_count)
                st["densities"].append(self._peak_density)
                st["speeds"].append(speed)
                st["direction_variances"].append(dir_var)
                st["risk_scores"].append(risk * 100.0)
                st["features"].append([D, S, V])
                st["risk_unit"].append(risk)
                st["fps_values"].append(fps)
                self.frames_processed += 1

                csv_w.writerow([
                    frame_index, round(frame_index / fps_video, 2),
                    smooth_count, round(self._peak_density, 3),
                    round(speed, 3), round(dir_var, 3),
                    round(raw, 4), round(risk, 4), level,
                ] + self.scene.csv_row())

                if self.show_heatmap:
                    frame = self._overlay_heatmap(frame, self._dm)

                # zone outline (and vehicle boxes if enabled)
                self.scene.draw(frame)

                white = (255, 255, 255)
                lines = [
                    (self.client_id, white),
                    (f"Scene: {self.scene.label}", white),
                    (f"People: {smooth_count}", white),
                    (f"Peak density: {self._peak_density:.2f} /m2", white),
                    (f"Speed: {speed:.2f} m/s", white),
                    (f"Direction var: {dir_var:.2f}", white),
                ]
                lines += self.scene.overlay_lines()
                lines.append((f"RISK: {risk * 100:.0f}/100 ({level})", color))
                lines += [(a, (60, 60, 255)) for a in alerts]
                lines.append((f"FPS: {fps_avg:.1f}", white))

                cv2.rectangle(frame, (5, 5), (420, 12 + 27 * len(lines)),
                              (35, 35, 35), -1)

                for i, (text, text_color) in enumerate(lines):
                    cv2.putText(
                        frame, text, (15, 30 + 27 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2,
                    )

                # status bar with the simulated control actions
                if self.controller is not None:
                    self.controller.draw_banner(frame)

                if writer is not None:
                    writer.write(frame)

                if self.display:
                    shown = frame
                    if show_size is not None:
                        shown = cv2.resize(
                            frame, show_size, interpolation=cv2.INTER_AREA
                        )

                    cv2.imshow(self.client_id, shown)

                    # Pace playback to the video's own FPS. If we are running
                    # behind, don't sleep and just reset the schedule.
                    if self.pace_to_video:
                        next_t += self.dt
                        delay = next_t - time.perf_counter()

                        if delay > 0:
                            wait_ms = max(1, int(delay * 1000))
                        else:
                            wait_ms = 1
                            next_t = time.perf_counter()
                    else:
                        wait_ms = 1

                    if self.controller is not None:
                        self.controller.show_window()

                    key = cv2.waitKey(wait_ms) & 0xFF

                    if self.controller is not None:
                        self.controller.handle_key(key)

                    if key == ord("q"):
                        break

                frame_index += 1

        finally:
            self._stop_worker()
            cap.release()
            csv_file.close()

            if self.controller is not None:
                self.controller.close()

            if writer is not None:
                writer.release()

            if self.display:
                cv2.destroyAllWindows()

        print(f"[{self.client_id}] Finished {frame_index} frames.")
        print(f"[{self.client_id}] Per-frame results: {csv_path}")

        return self.get_summary()

    # ------------------------------------------------------
    # LOCAL TRAINING (federated demo)
    # ------------------------------------------------------

    def simulate_local_training(self, epochs=30, target_label=None):
        """
        Train the small model to map [D, S, V] -> risk.
        target_label: optional constant label (e.g. 0.0 for a normal video,
        1.0 for a stampede video). If None, the formula risk is the target.
        """
        feats = self.local_stats["features"]

        if not feats:
            print(f"[{self.client_id}] No frames; skipping training.")
            return

        x = torch.tensor(feats, dtype=torch.float32)

        if target_label is None:
            y = torch.tensor(self.local_stats["risk_unit"],
                             dtype=torch.float32).unsqueeze(1)
        else:
            y = torch.full((len(feats), 1), float(target_label))

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.local_model.parameters(), lr=0.01)
        self.local_model.train()

        loss = None

        for _ in range(epochs):
            optimizer.zero_grad()
            loss = criterion(self.local_model(x), y)
            loss.backward()
            optimizer.step()

        print(
            f"[{self.client_id}] Local training complete "
            f"(final loss: {loss.item():.4f})"
        )

    def get_model_weights(self):
        return {
            k: v.detach().cpu().clone()
            for k, v in self.local_model.state_dict().items()
        }

    # ------------------------------------------------------
    # SUMMARIES
    # ------------------------------------------------------

    def get_risk_update(self):
        s = self.local_stats

        def avg(key):
            return float(np.mean(s[key])) if s[key] else 0.0

        return {
            "client_id": self.client_id,
            "frames": self.frames_processed,
            "avg_people": avg("person_counts"),
            "avg_density": avg("densities"),
            "avg_speed": avg("speeds"),
            "avg_direction_variance": avg("direction_variances"),
            "avg_risk": avg("risk_scores"),
            "max_risk": float(max(s["risk_scores"])) if s["risk_scores"] else 0.0,
            "avg_fps": avg("fps_values"),
            **self.scene.summary(),
        }

    def get_summary(self):
        u = self.get_risk_update()
        u["frames_processed"] = u["frames"]
        return u

    def print_summary(self):
        u = self.get_risk_update()
        print(f"{u['client_id']}:")
        print(f"  Frames processed : {u['frames']}")
        print(f"  Avg people       : {u['avg_people']:.1f}")
        print(f"  Avg peak density : {u['avg_density']:.2f} people/m2")
        print(f"  Avg speed        : {u['avg_speed']:.2f} m/s")
        print(f"  Avg dir variance : {u['avg_direction_variance']:.3f}")
        print(f"  Avg risk         : {u['avg_risk']:.1f}/100")
        print(f"  Max risk         : {u['max_risk']:.1f}/100")
        print(f"  Avg FPS          : {u['avg_fps']:.1f}")
        print(f"  Scene            : {self.scene.label}")

        if self.scene.track_vehicles:
            print(f"  Avg vehicles     : {u['avg_vehicles']:.1f} "
                  f"(occupancy {u['avg_vehicle_occupancy'] * 100:.0f}%)")

        if self.scene.zone_polygon is not None:
            print(f"  {self.scene.zone_label:<16} : avg "
                  f"{u['avg_zone_people']:.1f} / max "
                  f"{u['max_zone_people']:.1f} people")

        if self.controller is not None:
            c = self.controller.summary()
            print(f"  Control actions  : {c['actions_triggered']} triggered "
                  f"(highest level held: {c['max_level']})")
            print(f"  Action log       : {self.controller.log_path}")