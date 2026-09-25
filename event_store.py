"""Lightweight on-disk persistence for alerts, C2 events and reports.

This backs the "Data storage: pose sequences / alerts / reports" part of the
project, which previously lived only in memory (alerts were lost on restart).

Everything is append-only JSONL under `data/`, guarded by a lock so the video
thread and the Flask request threads can share it safely. Every write is
wrapped in try/except: a full or read-only disk must never take down the live
pipeline.
"""

import json
import os
import threading
from datetime import datetime

DEFAULT_BASE = "data"


class EventStore:
    def __init__(self, base_dir=DEFAULT_BASE):
        self.base_dir = base_dir
        self.alerts_dir = os.path.join(base_dir, "alerts")
        self.events_dir = os.path.join(base_dir, "events")
        self.reports_dir = os.path.join(base_dir, "reports")
        self.poses_dir = os.path.join(base_dir, "pose_sequences")
        for d in (self.alerts_dir, self.events_dir, self.reports_dir, self.poses_dir):
            try:
                os.makedirs(d, exist_ok=True)
            except Exception:
                pass

        self._lock = threading.Lock()
        self.alerts_path = os.path.join(self.alerts_dir, "alerts.jsonl")
        self.events_path = os.path.join(self.events_dir, "events.jsonl")
        self._warned = False

    # ------------------------------------------------------------------ helpers
    def _append_jsonl(self, path, payload):
        try:
            line = json.dumps(payload, default=str)
            with self._lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            return True
        except Exception as exc:
            if not self._warned:
                print(f"[EventStore] write failed for {path}: {exc}")
                self._warned = True
            return False

    # ------------------------------------------------------------------- writes
    def append_alert(self, alert: dict):
        return self._append_jsonl(self.alerts_path, alert)

    def append_event(self, event: dict):
        return self._append_jsonl(self.events_path, event)

    def append_pose_sequence(self, track_id, sequence_meta: dict):
        """Persist a compact pose-sequence record (metadata, not raw frames)."""
        record = {
            "track_id": track_id,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **sequence_meta,
        }
        return self._append_jsonl(
            os.path.join(self.poses_dir, "pose_sequences.jsonl"), record)

    def save_report(self, report: dict):
        """Write a report JSON to data/reports and return its path."""
        try:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.reports_dir, f"report_{stamp}.json")
            with self._lock:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2, default=str)
            return path
        except Exception as exc:
            print(f"[EventStore] report write failed: {exc}")
            return None

    # -------------------------------------------------------------------- reads
    def load_recent_alerts(self, limit=50):
        if not os.path.exists(self.alerts_path):
            return []
        out = []
        try:
            with self._lock:
                with open(self.alerts_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                out.append(json.loads(line))
                            except Exception:
                                continue
            return out[-limit:][::-1]
        except Exception:
            return []

    def list_reports(self, limit=20):
        try:
            files = sorted(
                (f for f in os.listdir(self.reports_dir) if f.endswith(".json")),
                reverse=True)
            return files[:limit]
        except Exception:
            return []

    def stats(self):
        def count(path):
            if not os.path.exists(path):
                return 0
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return sum(1 for line in f if line.strip())
            except Exception:
                return 0
        return {
            "alerts_logged": count(self.alerts_path),
            "events_logged": count(self.events_path),
            "reports_saved": len(self.list_reports(limit=1000)),
        }
