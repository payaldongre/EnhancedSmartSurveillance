"""Virtual fence / restricted-zone intrusion detection.

Completes the "Virtual fence intrusion detection" objective.

Zones are polygons stored in a JSON config file (default `zones.json`) using
resolution-independent NORMALISED coordinates (0.0-1.0 of frame width/height),
so the same file works no matter what camera resolution you run at.

Per tracked object the manager emits three event kinds:
    intrusion — object entered a restricted zone
    dwell     — object has stayed inside past the zone's dwell time (loitering)
    exit      — object left the zone

State is kept per (track_id, zone) so repeated frames inside the zone do not
re-fire the alert — same de-duplication philosophy as the weapon alerts.
"""

import json
import os
import time

import cv2
import numpy as np

DEFAULT_CONFIG = "zones.json"

# Distinct outline colours cycled per zone.
_ZONE_COLORS = [(0, 0, 255), (0, 200, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0)]


def _point_in_polygon(pt, polygon) -> bool:
    if not polygon or len(polygon) < 3:
        return False
    contour = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), False) >= 0


class VirtualFence:
    def __init__(self, config_path=None, enabled=True, default_dwell_seconds=5.0):
        self.config_path = config_path or os.environ.get("ZONES_CONFIG", DEFAULT_CONFIG)
        self.enabled = enabled
        self.default_dwell_seconds = default_dwell_seconds

        self.zones = []
        self.load()

        # track_id -> { zone_name: {"inside": bool, "enter_time": float, "dwell_fired": bool} }
        self.track_state = {}
        self.intrusions = 0
        self.dwell_events = 0

    # ---------------------------------------------------------------- config
    def load(self):
        if not os.path.exists(self.config_path):
            self.zones = []
            return
        try:
            with open(self.config_path, "r") as f:
                data = json.load(f)
            zones = data.get("zones", data if isinstance(data, list) else [])
            self.zones = [z for z in zones if self._valid_zone(z)]
        except Exception as exc:
            print(f"[VirtualFence] Could not read {self.config_path}: {exc}")
            self.zones = []

    def save(self):
        try:
            payload = {
                "note": "Normalised coordinates (0-1) relative to frame width/height.",
                "zones": self.zones,
            }
            with open(self.config_path, "w") as f:
                json.dump(payload, f, indent=2)
            return True
        except Exception as exc:
            print(f"[VirtualFence] Could not write {self.config_path}: {exc}")
            return False

    @staticmethod
    def _valid_zone(zone) -> bool:
        return (isinstance(zone, dict) and zone.get("name")
                and isinstance(zone.get("points"), list) and len(zone["points"]) >= 3)

    # ----------------------------------------------------------------- mutate
    def add_zone(self, name, points, dwell_seconds=None, kind="restricted", persist=True):
        zone = {
            "name": name,
            "points": [[float(p[0]), float(p[1])] for p in points],
            "kind": kind,
            "dwell_seconds": float(dwell_seconds) if dwell_seconds else self.default_dwell_seconds,
        }
        self.zones = [z for z in self.zones if z["name"] != name] + [zone]
        if persist:
            self.save()
        return zone

    def remove_zone(self, name):
        before = len(self.zones)
        self.zones = [z for z in self.zones if z["name"] != name]
        removed = before != len(self.zones)
        if removed:
            self.save()
        return removed

    def list_zones(self):
        return self.zones

    # --------------------------------------------------------------- runtime
    def evaluate(self, track_id, point_norm, label="person"):
        """Check one object position against every zone. Returns a list of events."""
        if not self.enabled or not self.zones:
            return []

        events = []
        state = self.track_state.setdefault(track_id, {})
        now = time.time()

        for zone in self.zones:
            name = zone["name"]
            inside = _point_in_polygon(point_norm, zone["points"])
            zstate = state.setdefault(
                name, {"inside": False, "enter_time": 0.0, "dwell_fired": False})

            if inside and not zstate["inside"]:
                zstate.update(inside=True, enter_time=now, dwell_fired=False)
                self.intrusions += 1
                events.append({
                    "type": "intrusion", "zone": name, "track_id": track_id,
                    "label": label, "kind": zone.get("kind", "restricted"),
                })
            elif inside and zstate["inside"]:
                dwell = zone.get("dwell_seconds", self.default_dwell_seconds)
                if not zstate["dwell_fired"] and (now - zstate["enter_time"]) >= dwell:
                    zstate["dwell_fired"] = True
                    self.dwell_events += 1
                    events.append({
                        "type": "dwell", "zone": name, "track_id": track_id,
                        "label": label, "kind": zone.get("kind", "restricted"),
                        "seconds": round(now - zstate["enter_time"], 1),
                    })
            elif not inside and zstate["inside"]:
                zstate.update(inside=False, enter_time=0.0, dwell_fired=False)
                events.append({
                    "type": "exit", "zone": name, "track_id": track_id,
                    "label": label, "kind": zone.get("kind", "restricted"),
                })

        return events

    def drop_tracks(self, active_ids):
        """Forget state for tracks that are no longer being tracked."""
        for tid in list(self.track_state.keys()):
            if tid not in active_ids:
                self.track_state.pop(tid, None)

    # ------------------------------------------------------------------ draw
    def draw(self, frame):
        if not self.enabled or frame is None:
            return frame
        h, w = frame.shape[:2]
        overlay = frame.copy()
        for i, zone in enumerate(self.zones):
            pts = np.array(
                [[int(p[0] * w), int(p[1] * h)] for p in zone["points"]], dtype=np.int32)
            color = _ZONE_COLORS[i % len(_ZONE_COLORS)]
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(frame, [pts], True, color, 2)
            x, y = pts[0]
            cv2.putText(frame, f"ZONE: {zone['name']}", (x + 5, max(18, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
        return frame

    def reset_stats(self):
        self.intrusions = 0
        self.dwell_events = 0
        self.track_state.clear()

    def stats(self):
        return {
            "enabled": self.enabled,
            "zones": [z["name"] for z in self.zones],
            "zone_count": len(self.zones),
            "intrusions": self.intrusions,
            "dwell_events": self.dwell_events,
        }
