"""Command & Control (C2) integration.

Completes the "Integration with existing command & control (C2) systems"
objective.

Events are normalised into one stable schema and pushed to any external system
that can consume HTTP webhooks (and optionally MQTT), which is the common
denominator for C2 / VMS / PSIM platforms:

    {
      "schema": "smart-surveillance.event/1.0",
      "event_id": "<uuid>",
      "timestamp": "2026-01-01T12:00:00Z",
      "source": {"system": "...", "camera_id": "Cam_1", "site": "..."},
      "type": "Fall Detected",
      "severity": "critical",
      "confidence": 0.95,
      "details": "human readable summary",
      "data": { ...type specific extras... }
    }

Design notes:
  * Publishing never blocks the video loop — events go onto a bounded queue
    drained by a daemon worker thread.
  * Uses only the standard library (urllib) for HTTP, so no new dependency is
    required for the webhook path. MQTT is optional (paho-mqtt).
  * Failures are retried with backoff and counted; nothing raises into the
    pipeline.

Configuration is entirely environment based:
  C2_ENABLED        "1"/"true" to enable (auto-enabled when a URL is set)
  C2_WEBHOOK_URL    destination endpoint
  C2_API_KEY        optional bearer token
  C2_SOURCE_ID      site/system identifier (default "site-1")
  C2_MQTT_HOST      optional MQTT broker host (enables MQTT when set)
  C2_MQTT_PORT      optional (default 1883)
  C2_MQTT_TOPIC     optional topic template (default surveillance/<source>/<camera>/events)
"""

import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone

SCHEMA_VERSION = "smart-surveillance.event/1.0"

# Which severities each alert type maps to on the C2 side.
_SEVERITY = {
    "Fall Detected": "critical",
    "Weapon Detected": "critical",
    "Intrusion Detected": "high",
    "Restricted Zone Dwell": "high",
    "Night-time Movement": "high",
    "Running Detected": "medium",
    "Abnormal Behavior": "medium",
    "ANPR Plate Read": "info",
    "Vehicle Detected": "info",
    "Face Detected": "info",
    "Test Alert": "info",
}


def severity_for(alert_type: str) -> str:
    return _SEVERITY.get(alert_type, "low")


def build_event(alert_type, confidence=0.0, camera_id="Cam_1", details="",
                source_id=None, data=None) -> dict:
    """Build a schema-compliant event dict."""
    return {
        "schema": SCHEMA_VERSION,
        "event_id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {
            "system": "EnhancedSmartSurveillance",
            "camera_id": camera_id,
            "site": source_id or os.environ.get("C2_SOURCE_ID", "site-1"),
        },
        "type": alert_type,
        "severity": severity_for(alert_type),
        "confidence": round(float(confidence or 0.0), 3),
        "details": details or "",
        "data": data or {},
    }


class C2Client:
    def __init__(self, webhook_url=None, api_key=None, source_id=None,
                 timeout=5.0, max_retries=3, queue_size=1000):
        self.webhook_url = webhook_url or os.environ.get("C2_WEBHOOK_URL", "").strip()
        self.api_key = api_key or os.environ.get("C2_API_KEY", "").strip()
        self.source_id = source_id or os.environ.get("C2_SOURCE_ID", "site-1")
        self.timeout = timeout
        self.max_retries = max_retries

        enabled_env = os.environ.get("C2_ENABLED", "").strip().lower()
        self.enabled = enabled_env in ("1", "true", "yes", "on") or bool(self.webhook_url)

        self.queue = queue.Queue(maxsize=queue_size)
        self.sent = 0
        self.failed = 0
        self.dropped = 0
        self.last_error = ""
        self.last_sent_at = None

        # Optional MQTT transport.
        self.mqtt_client = None
        self.mqtt_error = ""
        self._init_mqtt()

        self._worker = None
        if self.enabled:
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()
            transport = "webhook" if self.webhook_url else "none"
            if self.mqtt_client:
                transport += "+mqtt"
            print(f"[C2] Integration enabled ({transport}) -> {self.webhook_url or 'mqtt only'}")

    # -------------------------------------------------------------- transports
    def _init_mqtt(self):
        host = os.environ.get("C2_MQTT_HOST", "").strip()
        if not host:
            return
        try:
            import paho.mqtt.client as mqtt
            port = int(os.environ.get("C2_MQTT_PORT", "1883"))
            client = mqtt.Client(client_id=f"surveillance-{self.source_id}")
            if os.environ.get("C2_MQTT_USER"):
                client.username_pw_set(
                    os.environ["C2_MQTT_USER"], os.environ.get("C2_MQTT_PASS", ""))
            client.connect(host, port, keepalive=30)
            client.loop_start()
            self.mqtt_client = client
        except Exception as exc:
            self.mqtt_error = str(exc)
            print(f"[C2] MQTT unavailable: {exc}")

    def _publish_mqtt(self, event):
        if not self.mqtt_client:
            return False
        try:
            topic = os.environ.get(
                "C2_MQTT_TOPIC",
                f"surveillance/{self.source_id}/{event['source']['camera_id']}/events")
            self.mqtt_client.publish(topic, json.dumps(event))
            return True
        except Exception as exc:
            self.mqtt_error = str(exc)
            return False

    def _post_webhook(self, event):
        if not self.webhook_url:
            return False
        body = json.dumps(event).encode("utf-8")
        req = urllib.request.Request(self.webhook_url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", "EnhancedSmartSurveillance/2.0")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return 200 <= resp.status < 300

    # ------------------------------------------------------------------- public
    def publish(self, event) -> bool:
        """Queue an event for delivery. Never blocks the caller."""
        if not self.enabled:
            return False
        try:
            self.queue.put_nowait(event)
            return True
        except queue.Full:
            self.dropped += 1
            self.last_error = "queue full"
            return False

    def _deliver(self, event):
        ok = False
        if self.webhook_url:
            ok = self._post_webhook(event)
        if self.mqtt_client:
            ok = self._publish_mqtt(event) or ok
        return ok

    def _run(self):
        while True:
            event = self.queue.get()
            try:
                for attempt in range(1, self.max_retries + 1):
                    try:
                        if self._deliver(event):
                            self.sent += 1
                            self.last_sent_at = time.time()
                            break
                    except Exception as exc:
                        self.last_error = str(exc)
                    if attempt < self.max_retries:
                        time.sleep(min(2 ** attempt, 8) * 0.25)
                else:
                    self.failed += 1
            finally:
                self.queue.task_done()

    def send_test(self):
        return self.publish(build_event(
            "Test Alert", 1.0, "Cam_1", "C2 connectivity test", self.source_id))

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "transport": ("webhook" if self.webhook_url else "") + ("+mqtt" if self.mqtt_client else ""),
            "endpoint": self.webhook_url or None,
            "queued": self.queue.qsize(),
            "sent": self.sent,
            "failed": self.failed,
            "dropped": self.dropped,
            "last_error": self.last_error,
            "mqtt": bool(self.mqtt_client),
        }
