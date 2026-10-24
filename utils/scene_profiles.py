"""
utils/scene_profiles.py

Scene profiles: per-camera settings that change WHAT is measured and WHEN
an alert is raised, depending on where the camera is looking
(road, railway platform, event venue, ...).

IMPORTANT
---------
Every number in SCENE_PROFILES is a PLACEHOLDER that demonstrates the idea.
Replace them with limits from your site's risk assessment, the venue licence
or local authority guidance (for example NDMA / state crowd-management
guidelines). They are NOT legal standards.

Add your own scene by adding another entry to SCENE_PROFILES.
"""

import cv2
import numpy as np

# COCO class ids used by YOLO: bicycle, car, motorcycle, bus, truck
VEHICLE_CLASSES = [1, 2, 3, 5, 7]

LEVEL_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
LEVEL_COLOR = {
    "LOW": (0, 255, 0),
    "MEDIUM": (0, 165, 255),
    "HIGH": (0, 0, 255),
}

SCENE_PROFILES = {

    # General open crowd area (default). No extra analysis.
    "public_street": {
        "label": "Public street",
        "density_critical": 5.0,      # people/m2 that maps to density score 1.0
        "speed_critical": 2.5,        # m/s that maps to speed score 1.0
        "low_t": 0.35,                # risk below this = LOW
        "high_t": 0.65,               # risk above this = HIGH
        "track_vehicles": False,
        "zone_label": None,
        "zone_people_alert": None,    # people inside the zone that raise an alert
    },

    # Road: pedestrians are at risk at LOW density, vehicles are tracked.
    # zone_polygon = the roadway / carriageway.
    "road_traffic": {
        "label": "Road / traffic",
        "density_critical": 2.0,
        "speed_critical": 2.5,
        "low_t": 0.25,
        "high_t": 0.50,
        "track_vehicles": True,
        "vehicle_occupancy_high": 0.35,   # fraction of roadway covered by vehicles
        "zone_label": "Roadway",
        "zone_people_alert": 2,
        "zone_with_vehicles_is_high": True,
    },

    # Platform: the danger is the strip next to the track.
    # zone_polygon = the platform-edge strip.
    "railway_platform": {
        "label": "Railway platform",
        "density_critical": 3.0,
        "speed_critical": 2.0,
        "low_t": 0.30,
        "high_t": 0.55,
        "track_vehicles": False,
        "zone_label": "Platform edge",
        "zone_people_alert": 6,
    },

    # Stadium / concert / festival.
    "event_venue": {
        "label": "Event venue",
        "density_critical": 4.0,
        "speed_critical": 2.0,
        "low_t": 0.30,
        "high_t": 0.60,
        "track_vehicles": False,
        "zone_label": None,
        "zone_people_alert": None,
    },

    # Temple / religious gathering with narrow entrances.
    # zone_polygon (optional) = the entrance or bottleneck.
    "religious_gathering": {
        "label": "Religious gathering",
        "density_critical": 4.0,
        "speed_critical": 1.8,
        "low_t": 0.30,
        "high_t": 0.55,
        "track_vehicles": False,
        "zone_label": "Entrance",
        "zone_people_alert": 15,
    },
}


def escalate(level, minimum):
    """Return whichever of the two alert levels is more severe."""
    return level if LEVEL_RANK[level] >= LEVEL_RANK[minimum] else minimum


class SceneAnalyzer:
    """
    Holds one camera's scene profile and the extra measurements it needs:
        - vehicle count + road occupancy (YOLO11, road scenes only)
        - number of people inside a restricted zone (from the CSRNet map)
        - scene-specific alerts that can raise the alert level
    """

    def __init__(
        self,
        scene,
        zone_polygon=None,
        vehicle_model=None,
        vehicle_conf=0.30,
        draw_vehicles=False,
    ):
        if scene not in SCENE_PROFILES:
            raise ValueError(
                f"Unknown scene '{scene}'. "
                f"Choose from: {', '.join(SCENE_PROFILES)}"
            )

        self.name = scene
        self.profile = SCENE_PROFILES[scene]
        self.label = self.profile["label"]

        self.density_critical = self.profile["density_critical"]
        self.speed_critical = self.profile["speed_critical"]
        self.low_t = self.profile["low_t"]
        self.high_t = self.profile["high_t"]

        self.zone_label = self.profile.get("zone_label") or "Zone"
        self.zone_polygon = None

        if zone_polygon:
            self.zone_polygon = [(float(x), float(y)) for x, y in zone_polygon]

            if len(self.zone_polygon) < 3:
                raise ValueError("zone_polygon needs at least 3 points.")

        self.track_vehicles = bool(self.profile.get("track_vehicles"))
        self.vehicle_model = vehicle_model if self.track_vehicles else None
        self.vehicle_conf = vehicle_conf
        self.draw_vehicles = draw_vehicles

        if self.track_vehicles and vehicle_model is None:
            print(f"[{scene}] WARNING: no vehicle model given; "
                  f"vehicle tracking is disabled.")

        if (self.profile.get("zone_people_alert") is not None
                and self.zone_polygon is None):
            print(f"[{scene}] WARNING: no zone_polygon set; "
                  f"'{self.zone_label}' alerts are disabled. "
                  f"Use pick_zone.py to draw one.")

        # Latest measurements
        self.vehicles = []
        self.vehicle_count = 0
        self.vehicle_occupancy = 0.0
        self.zone_people = 0.0

        self._zone_cache = {}
        self._hist_vehicles = []
        self._hist_occupancy = []
        self._hist_zone = []

    # ------------------------------------------------------
    # VEHICLES (YOLO11)
    # ------------------------------------------------------

    def detect_vehicles(self, frame):
        """Returns a list of (x1, y1, x2, y2) vehicle boxes in frame pixels."""
        if self.vehicle_model is None:
            return []

        result = self.vehicle_model.predict(
            source=frame,
            classes=VEHICLE_CLASSES,
            conf=self.vehicle_conf,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []

        return [
            tuple(float(v) for v in box)
            for box in result.boxes.xyxy.cpu().numpy()
        ]

    # ------------------------------------------------------
    # ZONE HELPERS
    # ------------------------------------------------------

    def _zone_mask(self, shape):
        """Binary mask (uint8, 0/1) of the zone polygon at (h, w)."""
        if shape not in self._zone_cache:
            h, w = shape
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(
                [[x * w, y * h] for x, y in self.zone_polygon]
            ).round().astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
            self._zone_cache[shape] = mask

        return self._zone_cache[shape]

    # ------------------------------------------------------
    # UPDATE (called whenever a fresh CSRNet result arrives)
    # ------------------------------------------------------

    def update(self, dm, count, vehicles, frame_size, flow_size):
        """
        dm         : 2D density map from CSRNet
        count      : estimated people in the frame
        vehicles   : boxes from detect_vehicles()
        frame_size : (width, height) of the original frame
        flow_size  : (width, height) of the small analysis grid
        """
        self.vehicles = vehicles or []
        self.vehicle_count = len(self.vehicles)

        # --- people inside the zone (share of density-map mass x count) ---
        self.zone_people = 0.0

        if self.zone_polygon is not None:
            d = np.clip(dm, 0, None)
            total = float(d.sum())

            if total > 1e-6:
                mask = self._zone_mask(d.shape)
                fraction = float((d * mask).sum()) / total
                self.zone_people = float(count) * fraction

        # --- vehicle occupancy: share of the road covered by vehicles ---
        self.vehicle_occupancy = 0.0

        if self.track_vehicles:
            fw, fh = flow_size
            W, H = frame_size
            veh = np.zeros((fh, fw), dtype=np.uint8)

            for x1, y1, x2, y2 in self.vehicles:
                cv2.rectangle(
                    veh,
                    (int(x1 * fw / W), int(y1 * fh / H)),
                    (int(x2 * fw / W), int(y2 * fh / H)),
                    1, -1,
                )

            if self.zone_polygon is not None:
                zone = self._zone_mask((fh, fw))
                area = float(zone.sum())

                if area > 0:
                    self.vehicle_occupancy = float((veh & zone).sum()) / area
            else:
                self.vehicle_occupancy = float(veh.mean())

    # ------------------------------------------------------
    # ALERT RULES
    # ------------------------------------------------------

    def apply_rules(self, level, color):
        """
        Applies the scene's rules on top of the crowd risk level.
        Returns (level, color, alerts) where alerts is a list of strings.
        """
        p = self.profile
        alerts = []

        limit = p.get("zone_people_alert")

        if (limit is not None and self.zone_polygon is not None
                and self.zone_people >= limit):
            alerts.append(
                f"{self.zone_label}: ~{self.zone_people:.0f} people"
            )
            level = escalate(level, "MEDIUM")

            if p.get("zone_with_vehicles_is_high") and self.vehicle_count > 0:
                alerts.append("People in roadway WITH traffic")
                level = escalate(level, "HIGH")

        occ_limit = p.get("vehicle_occupancy_high")

        if (self.track_vehicles and occ_limit is not None
                and self.vehicle_occupancy >= occ_limit):
            alerts.append(
                f"Heavy traffic ({self.vehicle_occupancy * 100:.0f}% occupied)"
            )
            level = escalate(level, "MEDIUM")

        return level, LEVEL_COLOR[level], alerts

    # ------------------------------------------------------
    # LOGGING / SUMMARY
    # ------------------------------------------------------

    def record(self):
        self._hist_vehicles.append(self.vehicle_count)
        self._hist_occupancy.append(self.vehicle_occupancy)
        self._hist_zone.append(self.zone_people)

    def summary(self):
        def avg(values):
            return float(np.mean(values)) if values else 0.0

        return {
            "scene": self.name,
            "avg_vehicles": avg(self._hist_vehicles),
            "avg_vehicle_occupancy": avg(self._hist_occupancy),
            "avg_zone_people": avg(self._hist_zone),
            "max_zone_people": (
                float(max(self._hist_zone)) if self._hist_zone else 0.0
            ),
        }

    def csv_header(self):
        return ["scene", "vehicles", "vehicle_occupancy", "zone_people"]

    def csv_row(self):
        return [
            self.name,
            self.vehicle_count,
            round(self.vehicle_occupancy, 3),
            round(self.zone_people, 2),
        ]

    # ------------------------------------------------------
    # DISPLAY
    # ------------------------------------------------------

    def overlay_lines(self):
        """Extra (text, colour) lines for the dashboard panel."""
        white = (255, 255, 255)
        lines = []

        if self.track_vehicles:
            lines.append((
                f"Vehicles: {self.vehicle_count} "
                f"(occupancy {self.vehicle_occupancy * 100:.0f}%)",
                white,
            ))

        if self.zone_polygon is not None:
            lines.append((
                f"{self.zone_label}: ~{self.zone_people:.0f} people",
                white,
            ))

        return lines

    def draw(self, frame):
        """Draws the zone outline (and vehicle boxes if enabled)."""
        h, w = frame.shape[:2]

        if self.zone_polygon is not None:
            pts = np.array(
                [[x * w, y * h] for x, y in self.zone_polygon]
            ).round().astype(np.int32)

            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
            cv2.putText(
                frame, self.zone_label,
                (int(pts[0][0]), max(20, int(pts[0][1]) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
            )

        if self.draw_vehicles:
            for x1, y1, x2, y2 in self.vehicles:
                cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)),
                    (0, 165, 255), 1,
                )