"""
utils/regulation.py

SIMULATED regulation controller.

It turns the monitoring result (alert level + scene) into control actions:
signs, announcements, signal requests, entry-gate holds, staff alerts.
Nothing here talks to real hardware. Each action only changes a simulated
device state that is shown on screen and written to an action log, so you
can demonstrate and test the logic safely.

Design rules
------------
1. Hysteresis: a level must persist before actions fire (escalate_after_s)
   and must stay lower before they are released (release_after_s). This
   stops actions flickering on and off. Times use VIDEO time, not wall time.
2. Operator approval: high-impact actions (signals, gates) wait as PENDING
   until an operator presses A (approve) or X (reject). Use approval="auto"
   for headless / unattended demo runs.
3. Never block exits: actions may only restrict INFLOW or add capacity.
   Closing or locking an exit can make a crowd crush much worse, so those
   actions are forbidden and checked at import time.
4. Every event is written to a CSV log with its reason.

All ladders below are PLACEHOLDERS. Real deployments must follow the
operator's procedures and be approved by the responsible authority.
"""

import csv
import os

import cv2
import numpy as np

LEVEL_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
LEVEL_COLOR = {
    "LOW": (0, 255, 0),
    "MEDIUM": (0, 165, 255),
    "HIGH": (0, 0, 255),
}

# ------------------------------------------------------------
# ACTION CATALOGUE
#   device   : the simulated device this action drives
#   on       : the state that device shows while the action is ACTIVE
#   approval : True = needs an operator to approve first
# ------------------------------------------------------------
ACTIONS = {
    "advisory_announcement": {
        "label": "Advisory announcement",
        "device": "Public address", "on": "ADVISORY PLAYING",
        "approval": False,
    },
    "platform_announcement": {
        "label": "Announcement: stand behind the line",
        "device": "Public address", "on": "STAND BEHIND LINE",
        "approval": False,
    },
    "ped_warning_sign": {
        "label": "Pedestrian warning sign ON",
        "device": "Warning sign", "on": "ON",
        "approval": False,
    },
    "ped_clear_phase": {
        "label": "Request pedestrian-clear signal phase",
        "device": "Traffic signal", "on": "PED-CLEAR REQUESTED",
        "approval": True,
    },
    "hold_entry_gates": {
        "label": "Hold ENTRY gates (inflow only)",
        "device": "Entry gates", "on": "HELD",
        "approval": True,
    },
    "open_overflow_lanes": {
        "label": "Open overflow lanes",
        "device": "Overflow lanes", "on": "OPEN",
        "approval": False,
    },
    "alert_operator": {
        "label": "Alert control room",
        "device": "Control room", "on": "ALERTED",
        "approval": False,
    },
    "dispatch_staff": {
        "label": "Dispatch field staff",
        "device": "Field staff", "on": "DISPATCHED",
        "approval": False,
    },
}

DEFAULT_STATE = {
    "Public address": "IDLE",
    "Warning sign": "OFF",
    "Traffic signal": "NORMAL",
    "Entry gates": "OPEN",
    "Overflow lanes": "CLOSED",
    "Control room": "NORMAL",
    "Field staff": "STANDBY",
}

# ------------------------------------------------------------
# LADDER PER SCENE (HIGH also includes everything from MEDIUM)
# ------------------------------------------------------------
LADDERS = {
    "public_street": {
        "MEDIUM": ["advisory_announcement"],
        "HIGH": ["alert_operator", "dispatch_staff"],
    },
    "road_traffic": {
        "MEDIUM": ["ped_warning_sign"],
        "HIGH": ["alert_operator", "ped_clear_phase"],
    },
    "railway_platform": {
        "MEDIUM": ["platform_announcement"],
        "HIGH": ["alert_operator", "hold_entry_gates"],
    },
    "event_venue": {
        "MEDIUM": ["advisory_announcement", "open_overflow_lanes"],
        "HIGH": ["alert_operator", "hold_entry_gates", "dispatch_staff"],
    },
    "religious_gathering": {
        "MEDIUM": ["advisory_announcement", "open_overflow_lanes"],
        "HIGH": ["alert_operator", "hold_entry_gates", "dispatch_staff"],
    },
}

# Safety rule: actions that restrict EXITS must never exist in a ladder.
FORBIDDEN_ACTIONS = {"close_exit", "lock_exit", "block_exit"}

for _scene, _ladder in LADDERS.items():
    for _ids in _ladder.values():
        for _aid in _ids:
            if _aid in FORBIDDEN_ACTIONS:
                raise ValueError(
                    f"Scene '{_scene}' contains forbidden action '{_aid}'."
                )
            if _aid not in ACTIONS:
                raise ValueError(f"Unknown action '{_aid}' in '{_scene}'.")

FONT = cv2.FONT_HERSHEY_SIMPLEX


class RegulationController:

    def __init__(
        self,
        scene,
        log_path,
        title="Control panel (simulated)",
        approval="operator",
        escalate_after_s=3.0,
        release_after_s=10.0,
    ):
        if scene not in LADDERS:
            raise ValueError(
                f"No control ladder for scene '{scene}'. "
                f"Choose from: {', '.join(LADDERS)}"
            )

        if approval not in ("operator", "auto"):
            raise ValueError("approval must be 'operator' or 'auto'.")

        self.scene = scene
        self.ladder = LADDERS[scene]
        self.title = title
        self.approval = approval
        self.escalate_after_s = float(escalate_after_s)
        self.release_after_s = float(release_after_s)
        self.log_path = log_path

        self.level = "LOW"              # the level the controller is acting on
        self.max_level = "LOW"
        self.actions = {}               # action_id -> "ACTIVE" | "PENDING"
        self.triggered = 0

        self._rejected = set()
        self._up_since = None
        self._down_since = None
        self._t = 0.0
        self._frame = 0
        self._ctx = {}

        # devices shown in the panel, in ladder order
        self.devices = []
        for lvl in ("MEDIUM", "HIGH"):
            for aid in self.ladder.get(lvl, []):
                dev = ACTIONS[aid]["device"]
                if dev not in self.devices:
                    self.devices.append(dev)

        self._file = None
        self._writer = None

    # ------------------------------------------------------
    # LOGGING
    # ------------------------------------------------------

    def _ensure_open(self):
        if self._file is not None:
            return

        folder = os.path.dirname(self.log_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        self._file = open(self.log_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "time_s", "frame", "scene", "held_level",
            "action", "status", "reason",
        ])

    def _reason(self):
        c = self._ctx
        return (
            f"people={c.get('people', 0)}, "
            f"density={c.get('density', 0.0):.2f}/m2, "
            f"zone={c.get('zone', 0.0):.0f}, "
            f"risk={c.get('risk', 0.0):.0f}"
        )

    def _log(self, action, status):
        self._ensure_open()
        self._writer.writerow([
            round(self._t, 2), self._frame, self.scene, self.level,
            action, status, self._reason(),
        ])
        self._file.flush()

    # ------------------------------------------------------
    # CORE LOGIC
    # ------------------------------------------------------

    def _desired(self):
        rank = LEVEL_RANK[self.level]
        ids = []

        for lvl in ("MEDIUM", "HIGH"):
            if rank >= LEVEL_RANK[lvl]:
                for aid in self.ladder.get(lvl, []):
                    if aid not in ids:
                        ids.append(aid)

        return ids

    def _set_level(self, level):
        self.level = level

        if LEVEL_RANK[level] > LEVEL_RANK[self.max_level]:
            self.max_level = level

        self._log("-", "LEVEL_CHANGE")

    def _sync_actions(self):
        desired = self._desired()

        for aid in desired:
            if aid in self.actions or aid in self._rejected:
                continue

            needs_approval = (
                ACTIONS[aid]["approval"] and self.approval == "operator"
            )

            self.actions[aid] = "PENDING" if needs_approval else "ACTIVE"
            self.triggered += 1
            self._log(
                aid, "PENDING_APPROVAL" if needs_approval else "ACTIVATED"
            )

        for aid in list(self.actions):
            if aid not in desired:
                self._log(aid, "RELEASED")
                del self.actions[aid]

        # a rejected action may be proposed again once it has been released
        self._rejected &= set(desired)

    def update(self, t, frame_index, level, context=None):
        """
        t           : video time in seconds
        frame_index : current frame number
        level       : "LOW" / "MEDIUM" / "HIGH" from the monitoring system
        context     : dict with people, density, zone, risk (for the log)
        """
        self._ensure_open()
        self._t, self._frame = t, frame_index
        self._ctx = context or {}

        target = LEVEL_RANK[level]
        current = LEVEL_RANK[self.level]

        if target > current:
            self._down_since = None

            if self._up_since is None:
                self._up_since = t

            if t - self._up_since >= self.escalate_after_s:
                self._set_level(level)
                self._up_since = None

        elif target < current:
            self._up_since = None

            if self._down_since is None:
                self._down_since = t

            if t - self._down_since >= self.release_after_s:
                self._set_level(level)
                self._down_since = None

        else:
            self._up_since = None
            self._down_since = None

        self._sync_actions()

    def handle_key(self, key):
        """A = approve pending actions, X = reject them."""
        pending = [a for a, s in self.actions.items() if s == "PENDING"]

        if not pending:
            return

        if key == ord("a"):
            for aid in pending:
                self.actions[aid] = "ACTIVE"
                self._log(aid, "APPROVED")

        elif key == ord("x"):
            for aid in pending:
                del self.actions[aid]
                self._rejected.add(aid)
                self._log(aid, "REJECTED")

    def pending(self):
        return [a for a, s in self.actions.items() if s == "PENDING"]

    def device_states(self):
        states = {d: DEFAULT_STATE[d] for d in self.devices}

        for aid, status in self.actions.items():
            if status == "ACTIVE":
                spec = ACTIONS[aid]
                states[spec["device"]] = spec["on"]

        return states

    def summary(self):
        return {
            "actions_triggered": self.triggered,
            "max_level": self.max_level,
        }

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    # ------------------------------------------------------
    # DISPLAY
    # ------------------------------------------------------

    def draw_banner(self, frame):
        """Compact status bar at the bottom of the video frame."""
        h, w = frame.shape[:2]
        states = self.device_states()

        active = [
            f"{dev}: {state}" for dev, state in states.items()
            if state != DEFAULT_STATE[dev]
        ]

        text = "CONTROL (sim): " + (
            " | ".join(active) if active else "no action"
        )

        rows = [(text, (0, 165, 255) if active else (200, 200, 200))]

        pending = self.pending()
        if pending:
            rows.append((
                f"APPROVAL NEEDED: {ACTIONS[pending[0]]['label']}"
                f"{' +' + str(len(pending) - 1) if len(pending) > 1 else ''}"
                f"   [A] approve  [X] reject",
                (0, 255, 255),
            ))

        bar_h = 26 * len(rows) + 6
        cv2.rectangle(frame, (0, h - bar_h), (w, h), (35, 35, 35), -1)

        for i, (line, color) in enumerate(rows):
            cv2.putText(
                frame, line, (10, h - bar_h + 22 + 26 * i),
                FONT, 0.5, color, 1,
            )

    def show_window(self):
        """Separate window with the full simulated device table."""
        states = self.device_states()
        pending = self.pending()

        lines = [
            ("SIMULATED CONTROLLER", (255, 255, 0)),
            (f"Scene: {self.scene}", (255, 255, 255)),
            (f"Acting on level: {self.level}", LEVEL_COLOR[self.level]),
            ("", (0, 0, 0)),
        ]

        for dev, state in states.items():
            changed = state != DEFAULT_STATE[dev]
            lines.append((
                f"{dev:<16} {state}",
                (0, 165, 255) if changed else (200, 200, 200),
            ))

        if pending:
            lines.append(("", (0, 0, 0)))

            for aid in pending:
                lines.append((f"PENDING: {ACTIONS[aid]['label']}",
                              (0, 255, 255)))

            lines.append(("Press A = approve   X = reject", (0, 255, 255)))

        height = 20 + 28 * len(lines)
        canvas = np.full((height, 560, 3), 35, dtype=np.uint8)

        for i, (text, color) in enumerate(lines):
            if text:
                cv2.putText(
                    canvas, text, (12, 30 + 28 * i), FONT, 0.55, color, 1,
                )

        cv2.imshow(self.title, canvas)