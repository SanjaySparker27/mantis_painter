#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import socket
import threading
import time
from dataclasses import dataclass

import cv2
import gz.transport13 as gz_transport
import numpy as np

# Limit thread fan-out: cv2 + numpy + torch (via env) all default to all cores
# which makes YOLO + draw_overlay starve the Flask/MJPEG threads. Pin them.
cv2.setNumThreads(1)
try:
    import os as _os
    _os.environ.setdefault("OMP_NUM_THREADS", "2")
    _os.environ.setdefault("MKL_NUM_THREADS", "2")
    _os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
except Exception:
    pass
from flask import Flask, Response, jsonify, request
from gz.msgs10.double_pb2 import Double
from gz.msgs10.image_pb2 import Image
from gz.msgs10.int32_pb2 import Int32
from gz.msgs10.model_pb2 import Model
from gz.msgs10.twist_pb2 import Twist


app = Flask(__name__)

PAN_LIMIT = (-85.29999907243065, 89.19999610737291)
TILT_LIMIT = (-39.99999883637168, 30.000000834826057)
CAMERA_TOPIC = "/mantis/nose_camera/image"
PAN_TOPIC = "/mantis/pan_cmd"
TILT_TOPIC = "/mantis/tilt_cmd"
JOINT_STATE_TOPIC = "/mantis/joint_states"
PAINT_TOPIC = "/mantis/paint_trigger"
PAINT_SIGNAL_FILE = "/tmp/mantis_paint.signal"
PAINT_PULSE_MS_DEFAULT = 120

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

# Paint output channels — each independent of the others. Any combination can
# be enabled at runtime so the MANTIS can drive a Pi GPIO via serial, an MCU
# via UDP, a network actuator via TCP, etc.
paint_channels = {
    "gz_topic": True,    # gz.msgs.Int32 on PAINT_TOPIC (default for sim)
    "file": True,        # append line to PAINT_SIGNAL_FILE
    "udp": False,
    "tcp": False,
    "serial": False,
}
paint_udp_addr = ("127.0.0.1", 9000)
paint_tcp_addr = ("127.0.0.1", 9001)
paint_serial_port = "/dev/ttyUSB0"
paint_serial_baud = 9600

agent_enabled = False
agent_model: str | None = None
agent_status = "idle"
agent_chat_log: list[dict] = []

IMG_W = 1280
IMG_H = 720
HFOV_RAD = 1.012
VFOV_RAD = 2.0 * math.atan(math.tan(HFOV_RAD / 2.0) * (IMG_H / IMG_W))

HOME_PAN_DEG = 0.0
HOME_TILT_DEG = 12.0
HOME_MAX_RATE_DEG_S = 35.0

# Pixel error sign conventions verified against pan_joint (+Z body yaw) and
# tilt_joint (+Y in pan_link). For a target right of center (ex>0) the body
# must yaw right => pan_deg must decrease, hence PAN_SIGN = -1. For a target
# below center (ey>0) the nose must tip down => tilt_deg must increase, hence
# TILT_SIGN = +1.
PAN_SIGN = -1.0
TILT_SIGN = +1.0


@dataclass
class Detection:
    det_id: int
    name: str
    bbox: tuple[int, int, int, int]
    score: float
    color: tuple[int, int, int]


@dataclass
class Gains:
    kp: float
    ki: float
    kd: float
    max_rate_deg_s: float
    integral_clamp_deg: float
    deadband_norm: float


# Verified by scripts/auto_tune_trial.py over multiple iterations: this set
# gave SS ex=+0.005±0.004 / ey=+0.006±0.006 (sub-pixel jitter on the car
# target) and lock in ~1.8 s with no oscillation or divergence at zoom=1.
pan_gains = Gains(kp=0.55, ki=0.20, kd=0.15, max_rate_deg_s=55.0,
                  integral_clamp_deg=4.0, deadband_norm=0.008)
tilt_gains = Gains(kp=0.55, ki=0.20, kd=0.15, max_rate_deg_s=40.0,
                   integral_clamp_deg=3.0, deadband_norm=0.012)


lock = threading.Lock()
node = gz_transport.Node()
pan_pub = node.advertise(PAN_TOPIC, Double)
tilt_pub = node.advertise(TILT_TOPIC, Double)
paint_pub = node.advertise(PAINT_TOPIC, Int32)
moving_target_pub = node.advertise("/moving_target/cmd_vel", Twist)
car_prius_pub = node.advertise("/car_prius_front/cmd_vel", Twist)
car_pickup_pub = node.advertise("/car_pickup_right/cmd_vel", Twist)
ball_red_pub = node.advertise("/ball_red/cmd_vel", Twist)
ball_green_pub = node.advertise("/ball_green/cmd_vel", Twist)
ring_bus_pub = node.advertise("/ring_bus_a/cmd_vel", Twist)

latest_raw: np.ndarray | None = None
latest_annotated: bytes | None = None
latest_stamp = 0.0
detections: list[Detection] = []
recent_detections: list[Detection] = []
recent_detection_stamp = 0.0
selected_id: int | None = None
selected_name: str | None = None
selected_anchor_xy: tuple[float, float] | None = None
pan_deg = 0.0
tilt_deg = 0.0
actual_pan_deg = 0.0
actual_tilt_deg = 12.0
pan_vel_deg_s = 0.0
tilt_vel_deg_s = 0.0
joint_state_stamp = 0.0
target_vx_pix_s = 0.0
target_vy_pix_s = 0.0
last_target_cx = 0.0
last_target_cy = 0.0
last_target_seen_ts = 0.0
smoothed_cx = 0.0
smoothed_cy = 0.0
smoothed_init = False
selected_signature: np.ndarray | None = None   # HSV histogram of the selected bbox (EMA-refreshed)
selected_signature_anchor: np.ndarray | None = None  # immutable snapshot captured at /api/select — used as fallback when EMA-blended signature has drifted away from the original target
_resolve_via_same_id: bool = False   # set True only when resolver returned via pre-flight same_id branch
# Pass 1 tightening thresholds (Fix A)
PASS1_SIG_MAX = 0.35       # Bhattacharyya cap (was 0.45)
PASS1_SIG_GAP = 0.05       # top vs 2nd-best sig_d must differ by at least this; only enforced when 3+ candidates
PASS1_IOU_MIN = 0.15       # candidate bbox must overlap KF-predicted bbox by this (skipped when pred_bbox unavailable)
# New-ID confirmation hysteresis (Fix C)
NEW_ID_CONFIRM_FRAMES = 3  # require this many consecutive resolver hits on a new det_id before rebinding selected_id
_pending_new_id: int | None = None
_pending_new_id_streak: int = 0
# Long-lost re-acquire tightening (Fix D)
LONGLOST_SIG_MAX = 0.38    # re-bind via long-lost path when sig distance under this (loosened so occluded target re-emerging with slight lighting change still matches)
# Auto-zoom (auto-mode): keep the locked target's bbox area inside a band.
auto_zoom_enabled: bool = False
AUTO_ZOOM_LOW = 0.012      # bbox area / frame area below this -> zoom in
AUTO_ZOOM_HIGH = 0.18      # bbox area / frame area above this -> zoom out
AUTO_ZOOM_STEP = 0.2
AUTO_ZOOM_COOLDOWN_S = 1.0
_last_auto_zoom_ts: float = 0.0


class TargetKF:
    """Constant-velocity Kalman filter on (cx, cy) bbox center in image
    pixels. State = [cx, cy, vx_px_s, vy_px_s]."""

    def __init__(self, cx: float, cy: float):
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=float)
        self.P = np.eye(4, dtype=float) * 200.0
        # Process noise — higher for velocity means we trust new
        # measurements more after sudden direction changes.
        self.Q = np.diag([4.0, 4.0, 80.0, 80.0]).astype(float)
        # Measurement noise — YOLO bbox center jitter ~4 px.
        self.R = np.diag([16.0, 16.0]).astype(float)
        self.last_ts = time.time()

    def predict(self, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        dt = max(1e-3, min(0.5, now - self.last_ts))
        self.last_ts = now
        F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, cx: float, cy: float) -> None:
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        z = np.array([cx, cy])
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    def position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    def velocity(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    def mahalanobis(self, cx: float, cy: float) -> float:
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        S = H @ self.P @ H.T + self.R
        z = np.array([cx, cy]) - H @ self.x
        try:
            Si = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return float("inf")
        return float(z @ Si @ z)


selected_kf: TargetKF | None = None
KF_MAHA_GATE = 25.0   # max Mahalanobis^2 distance for accepting a measurement

# Motion model in joint-bearing space — used to actively pursue the target
# through detection gaps instead of freezing the camera.
target_world_pan_deg: float | None = None
target_world_tilt_deg: float | None = None
target_pan_rate_deg_s: float = 0.0
target_tilt_rate_deg_s: float = 0.0
last_bearing_ts: float = 0.0
PURSUIT_DECAY_S = 1.8    # bearing-velocity decay constant during loss
PURSUIT_MAX_S = 4.0      # stop extrapolating after this long
last_target_w = 60.0
last_target_h = 60.0
PERSIST_HORIZON_S = 6.0  # forward-fill detection up to this long after a miss (longer = survives bigger occlusions)
LONG_LOST_REACQUIRE_S = 120.0  # try same-name re-acquisition for this long
pan_i_deg = 0.0
tilt_i_deg = 0.0
last_ex_norm = 0.0
last_ey_norm = 0.0
last_control_ts = 0.0
last_target_ts = 0.0
centered_frames = 0
virtual_marks: list[dict] = []
frame_count = 0
mode = "auto"  # auto | manual | home | stop
jog_pan_target: float | None = None
jog_tilt_target: float | None = None
detector_mode = "auto"  # auto = prefer YOLO+ByteTrack, color = force color
yolo_status = "init"
last_command_pan_deg = 0.0
last_command_tilt_deg = 12.0
zoom_factor = 1.0  # 1.0 = full FOV; >1 = digital zoom (crop+resize, narrower FOV)
ZOOM_MIN = 1.0
ZOOM_MAX = 6.0
zoom_changed_ts = 0.0
ZOOM_GRACE_S = 2.0

paint_count = 0
paint_last_ts = 0.0
paint_auto = False
paint_auto_min_centered = 5
paint_overlay_marks: list[dict] = []
PAINT_OVERLAY_TTL_S = 1.6

sweep_enabled = False
# Per-INSTANCE memory of which detection ids we've already painted in this
# sweep cycle. Sweep cycles through distinct det_ids (not names), so two
# cars on screen each get painted instead of "car class painted, skip".
sweep_painted_ids: set[int] = set()
# Backwards-compat alias surfaced via /api/status so the UI doesn't crash;
# count is what matters there.
sweep_painted_names: set[str] = set()
sweep_last_advance_ts = 0.0
center_hold_start_ts: float = 0.0
sweep_center_hold_start_ts: float = 0.0
SWEEP_PAINT_HOLD_S = 2.0      # require this much sustained centering before painting
SWEEP_CENTER_TOL_NORM = 0.12  # 12 % of frame from center counts as centered for the sweep painter (deadband is far tighter for the controller itself)
# Scanning: after painting, if no fresh target is in current FOV the sweep
# pans the camera back-and-forth between pan limits until a new (un-painted)
# detection appears. Stops the sweep from re-locking the same just-painted
# object frame after frame.
SWEEP_SCAN_RATE_DEG_S = 25.0
SWEEP_SCAN_MAX_S = 12.0     # after this long with no new target, reset painted memory
sweep_scan_dir: int = 1
sweep_scan_start_ts: float = 0.0
SWEEP_PER_TARGET_TIMEOUT_S = 5.0
SWEEP_MEMORY_FILE = "/tmp/mantis_painted_memory.json"


def _save_painted_memory():
    try:
        import json as _json
        with open(SWEEP_MEMORY_FILE, "w") as f:
            _json.dump(sorted(sweep_painted_names), f)
    except Exception:
        pass


def _load_painted_memory():
    global sweep_painted_names
    try:
        import json as _json, os as _os
        if _os.path.exists(SWEEP_MEMORY_FILE):
            with open(SWEEP_MEMORY_FILE) as f:
                sweep_painted_names = set(_json.load(f))
    except Exception:
        sweep_painted_names = set()


_load_painted_memory()

LOST_GRACE_S = 6.0  # hold selection through long detection gaps (don't drop to home)

YOLO_WEIGHTS_TRY = [
    "yolo12n.pt",
    "/home/sanju/yolo11n.pt",
    "yolo11n.pt",
    "yolov8n.pt",
]
_yolo_model = None
_yolo_tracker_cfg = "bytetrack_mantis.yaml"  # tuned for Mantis occlusion recovery
_yolo_class_palette = {}
_yoloworld_model = None
_yoloworld_classes: list[str] = ["car", "person", "truck", "drone"]
_yoloworld_classes_loaded: tuple[str, ...] = tuple()


def _load_yolo():
    global _yolo_model, yolo_status
    if _yolo_model is not None:
        return _yolo_model
    try:
        import torch  # type: ignore
        torch.set_num_threads(2)
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
    except Exception:
        pass
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        yolo_status = f"ultralytics import failed: {exc}"
        return None
    for w in YOLO_WEIGHTS_TRY:
        try:
            m = YOLO(w)
            _ = m.predict  # touch
            _yolo_model = m
            yolo_status = f"loaded {w}"
            return m
        except Exception as exc:
            yolo_status = f"load {w} failed: {exc}"
    return None


def _patch_clip_sha_check():
    """OpenAI CLIP hard-codes an Azure SHA in the URL path and rejects a
    cached file whose SHA doesn't match. The CDN has re-uploaded the same
    weights with a different SHA, so the bundled check always fails. We
    short-circuit it: if the cached file already exists, trust it."""
    try:
        import clip.clip as _cc
        orig = _cc._download
        def _trusting_download(url, root):
            import os
            target = os.path.join(root, os.path.basename(url))
            if os.path.exists(target) and os.path.getsize(target) > 1024 * 1024:
                return target
            return orig(url, root)
        _cc._download = _trusting_download
    except Exception:
        pass


def _load_yoloworld():
    """Lazy-load YOLO-World. Open-vocab detector keyed on free-text classes
    set via _yoloworld_classes. Same Ultralytics tracker API as v12."""
    global _yoloworld_model, yolo_status
    if _yoloworld_model is not None:
        return _yoloworld_model
    _patch_clip_sha_check()
    try:
        from ultralytics import YOLOWorld  # type: ignore
    except Exception as exc:
        yolo_status = f"YOLOWorld import failed: {exc}"
        return None
    for w in ("yolov8s-world.pt", "yolov8m-world.pt"):
        try:
            m = YOLOWorld(w)
            _ = m.predict
            _yoloworld_model = m
            yolo_status = f"loaded {w}"
            return m
        except Exception as exc:
            yolo_status = f"load {w} failed: {exc}"
    return None


def detect_with_yoloworld(frame: np.ndarray) -> list[Detection] | None:
    global _yoloworld_classes_loaded
    model = _load_yoloworld()
    if model is None:
        return None
    want = tuple(_yoloworld_classes)
    if want != _yoloworld_classes_loaded and want:
        try:
            model.set_classes(list(want))
            _yoloworld_classes_loaded = want
        except Exception as exc:
            globals()["yolo_status"] = f"set_classes failed: {exc}"
    try:
        results = model.track(
            source=frame,
            persist=True,
            tracker=_yolo_tracker_cfg,
            conf=0.08,
            iou=0.4,
            verbose=False,
        )
    except Exception as exc:
        globals()["yolo_status"] = f"world track exception: {exc}"
        return None
    out: list[Detection] = []
    if not results:
        return out
    res = results[0]
    boxes = getattr(res, "boxes", None)
    if boxes is None:
        return out
    # Map class index to the prompt the user typed. Ultralytics keeps
    # numeric labels in res.names after set_classes, so we look up our own
    # list instead.
    classes = list(_yoloworld_classes_loaded) or list(_yoloworld_classes)
    for i in range(len(boxes)):
        b = boxes[i]
        xyxy = b.xyxy[0].tolist()
        cls = int(b.cls[0]) if b.cls is not None else -1
        score = float(b.conf[0]) if b.conf is not None else 0.0
        det_id = int(b.id[0]) if (b.id is not None and len(b.id) > 0) else -1
        name = classes[cls] if 0 <= cls < len(classes) else str(cls)
        out.append(Detection(
            det_id=det_id, name=name,
            bbox=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
            score=score, color=_color_for_class(cls),
        ))
    return out


def _color_for_class(cid: int) -> tuple[int, int, int]:
    if cid in _yolo_class_palette:
        return _yolo_class_palette[cid]
    rng = np.random.default_rng(cid * 9173 + 17)
    c = tuple(int(x) for x in rng.integers(60, 230, size=3))
    _yolo_class_palette[cid] = c
    return c


def detect_with_yolo(frame: np.ndarray) -> list[Detection] | None:
    model = _load_yolo()
    if model is None:
        return None
    try:
        results = model.track(
            source=frame,
            persist=True,
            tracker=_yolo_tracker_cfg,
            conf=0.08,  # very permissive — let ByteTrack filter weak hits
            iou=0.4,
            imgsz=384,
            verbose=False,
        )
    except Exception as exc:
        global yolo_status
        yolo_status = f"track failed: {exc}"
        return None
    out: list[Detection] = []
    if not results:
        return out
    r0 = results[0]
    names = getattr(r0, "names", {}) or {}
    boxes = getattr(r0, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return out
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
    cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
    ids = boxes.id.cpu().numpy().astype(int) if (boxes.id is not None) else np.arange(1, len(xyxy) + 1)
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = (int(v) for v in xyxy[i])
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        cid = int(cls[i])
        name = names.get(cid, f"class_{cid}")
        out.append(Detection(
            det_id=int(ids[i]),
            name=str(name),
            bbox=(x1, y1, x2, y2),
            score=float(conf[i]),
            color=_color_for_class(cid),
        ))
    out.sort(key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
             reverse=True)
    return out[:16]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def safe_float(d: dict, key: str, default: float,
               lo: float = -1e9, hi: float = 1e9) -> float:
    try:
        v = float(d.get(key, default))
    except (TypeError, ValueError):
        v = default
    if not (lo <= v <= hi):
        v = clamp(v, lo, hi)
    return v


def safe_int(d: dict, key: str, default: int,
             lo: int = -10**9, hi: int = 10**9) -> int:
    try:
        v = int(d.get(key, default))
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


GAIN_BOUNDS = {
    # Kp > 1.0 overshoots the tilt axis on this rig — clamp here so a
    # sticky slider or accidental drag can't induce vertical bobble.
    "kp": (0.0, 1.0),
    "ki": (0.0, 1.0),
    "kd": (0.0, 1.0),
    "max_rate": (1.0, 80.0),
    "deadband": (0.0, 0.20),
}


def publish_angle(pub, deg: float) -> None:
    msg = Double()
    msg.data = math.radians(deg)
    pub.publish(msg)


def publish_pan_tilt() -> None:
    publish_angle(pan_pub, pan_deg)
    publish_angle(tilt_pub, tilt_deg)


def apply_digital_zoom(frame: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Crop the central region of the frame by `zoom_factor`, resize back to
    the original size, and return the (zoomed_frame, eff_hfov, eff_vfov).
    Detection + controller use the returned frame, so the system effectively
    has narrower FOV but higher pixel-per-degree at the cost of peripheral
    coverage.
    """
    z = clamp(zoom_factor, ZOOM_MIN, ZOOM_MAX)
    if z <= 1.001:
        return frame, HFOV_RAD, VFOV_RAD
    h, w = frame.shape[:2]
    cw = max(8, int(w / z))
    ch = max(8, int(h / z))
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    crop = frame[y0:y0 + ch, x0:x0 + cw]
    zoomed = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return zoomed, HFOV_RAD / z, VFOV_RAD / z


_eff_hfov = HFOV_RAD
_eff_vfov = VFOV_RAD


def image_to_bgr(msg: Image) -> np.ndarray | None:
    width = int(msg.width)
    height = int(msg.height)
    if width <= 0 or height <= 0 or not msg.data:
        return None
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    channels = max(1, int(len(arr) / max(1, width * height)))
    if channels < 3:
        return None
    arr = arr[: width * height * channels].reshape((height, width, channels))
    rgb = arr[:, :, :3]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def detect_colored_targets(frame: np.ndarray) -> list[Detection]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    specs = [
        ("red_box",      (0, 80, 60),     (10, 255, 255),  (40, 80, 255)),
        ("red_box",      (170, 80, 60),   (180, 255, 255), (40, 80, 255)),
        ("blue_box",     (95, 70, 50),    (130, 255, 255), (255, 140, 40)),
        ("green_box",    (45, 90, 70),    (70, 255, 255),  (80, 220, 100)),
        ("yellow_box",   (22, 130, 130),  (32, 255, 255),  (60, 220, 235)),
        ("orange_pillar",(11, 150, 150),  (20, 255, 255),  (40, 140, 250)),
        ("cyan_box",     (86, 90, 80),    (94, 255, 255),  (220, 220, 60)),
        ("magenta_box",  (140, 80, 70),   (168, 255, 255), (200, 60, 200)),
        ("lime_sphere",  (33, 120, 110),  (44, 255, 255),  (60, 240, 180)),
        ("teal_cone",    (78, 90, 70),    (88, 255, 255),  (190, 200, 60)),
        ("pink_disc",    (155, 70, 130),  (172, 255, 255), (175, 90, 235)),
        ("brown_block",  (8, 90, 50),     (18, 200, 130),  (40, 80, 130)),
        ("purple_pillar",(125, 90, 70),   (140, 255, 220), (180, 60, 130)),
    ]
    found: list[Detection] = []
    next_id = 1
    for name, lo, hi, color in specs:
        mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 450:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 18 or h < 18:
                continue
            found.append(Detection(next_id, name, (x, y, x + w, y + h),
                                   min(0.99, area / 30000.0), color))
            next_id += 1
    found.sort(key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
               reverse=True)
    for i, det in enumerate(found, 1):
        det.det_id = i
    return found[:12]


def reset_controller_state() -> None:
    global pan_i_deg, tilt_i_deg, last_ex_norm, last_ey_norm, centered_frames
    pan_i_deg = 0.0
    tilt_i_deg = 0.0
    last_ex_norm = 0.0
    last_ey_norm = 0.0
    centered_frames = 0


def clear_selection() -> None:
    global selected_id, selected_name, selected_anchor_xy
    global smoothed_init, target_vx_pix_s, target_vy_pix_s, last_target_seen_ts
    global selected_signature, selected_kf
    global target_world_pan_deg, target_world_tilt_deg
    global target_pan_rate_deg_s, target_tilt_rate_deg_s, last_bearing_ts
    global _pending_new_id, _pending_new_id_streak
    _pending_new_id = None
    _pending_new_id_streak = 0
    target_world_pan_deg = None
    target_world_tilt_deg = None
    target_pan_rate_deg_s = 0.0
    target_tilt_rate_deg_s = 0.0
    last_bearing_ts = 0.0
    selected_kf = None
    selected_id = None
    selected_name = None
    selected_anchor_xy = None
    selected_signature = None
    global selected_signature_anchor
    selected_signature_anchor = None
    smoothed_init = False
    target_vx_pix_s = 0.0
    target_vy_pix_s = 0.0
    last_target_seen_ts = 0.0
    reset_controller_state()
    # Do NOT reset ByteTrack tracker here — that would invalidate the IDs
    # the user just selected on the click that triggered this call.
    # Use clear_selection_full() for explicit deselects / mode changes.


def clear_selection_full() -> None:
    clear_selection()
    _bytetrack_reset()


def _bytetrack_reset() -> None:
    """Drop the persistent ByteTrack track list. New ones will be created on
    the next inference. Prevents stale-ID confusion + slow tracker bloat."""
    if _yolo_model is None:
        return
    try:
        preds = getattr(_yolo_model, "predictor", None)
        if preds and getattr(preds, "trackers", None):
            for t in preds.trackers:
                try:
                    t.reset()
                except Exception:
                    pass
    except Exception:
        pass


MAX_ANCHOR_REASSOC_PX = 160.0           # spinning car can shift bbox quite a bit
MAX_ANCHOR_REASSOC_CROSS_CLASS_PX = 100.0
BBOX_SIZE_RATIO_TOL = 0.40              # spinning car's bbox W/H change with viewing angle
MIN_TRACK_SCORE = 0.05                  # YOLO score dips below 0.10 mid-spin
AMBIGUITY_MARGIN = 1.25                 # was 1.6 — fewer false "too ambiguous" rejections


def _nearest_to_anchor(cands, ax, ay):
    return min(cands, key=lambda d: (
        ((d.bbox[0] + d.bbox[2]) / 2 - ax) ** 2
        + ((d.bbox[1] + d.bbox[3]) / 2 - ay) ** 2
    ))


def _hsv_signature_from_bbox(frame: np.ndarray, bbox) -> np.ndarray | None:
    if frame is None:
        return None
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0])); y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2])); y2 = min(h, int(bbox[3]))
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def _signature_distance(sig_a, sig_b) -> float:
    """Bhattacharyya distance: 0 = identical, 1 = totally different."""
    if sig_a is None or sig_b is None:
        return 1.0
    try:
        return float(cv2.compareHist(sig_a, sig_b, cv2.HISTCMP_BHATTACHARYYA))
    except Exception:
        return 1.0


def _candidate_score(d, ax, ay, ref_sig):
    """Composite score (lower = better) combining anchor distance + bbox-
    size similarity + HSV histogram match."""
    cx = (d.bbox[0] + d.bbox[2]) / 2
    cy = (d.bbox[1] + d.bbox[3]) / 2
    dist = math.hypot(cx - ax, cy - ay)
    cand_sig = _hsv_signature_from_bbox(latest_raw, d.bbox)
    sig_dist = _signature_distance(ref_sig, cand_sig)
    w = float(d.bbox[2] - d.bbox[0])
    h = float(d.bbox[3] - d.bbox[1])
    size_ratio = 1.0
    if last_target_w > 0 and last_target_h > 0:
        size_ratio = (
            min(w, last_target_w) / max(w, last_target_w)
            * min(h, last_target_h) / max(h, last_target_h)
        )
    # weights tuned so signature dominates when distances are similar
    return dist + 800.0 * sig_dist + 200.0 * (1.0 - size_ratio)


def _confirm_new_id(cand: "Detection") -> "Detection | None":
    """Hysteresis on selected_id rebind. When the resolver wants to bind onto
    a det_id different from selected_id, require NEW_ID_CONFIRM_FRAMES
    consecutive hits on that same new det_id first. Until confirmed, hand
    back the ghost so the controller holds the last-known position instead
    of jumping. Returns cand on confirm/same-id, ghost (or cand fallback if
    ghost expired) while pending."""
    global _pending_new_id, _pending_new_id_streak
    if selected_id is None or cand.det_id == selected_id:
        _pending_new_id = None
        _pending_new_id_streak = 0
        return cand
    # During Auto Serial Tracker centering, the camera is trying to hold
    # the bbox in the deadband for 2 s. Any rebind to a sibling here causes
    # the visible [LOCKED] box to shift mid-paint. Lock down hard: return
    # ghost until centering completes or the watchdog kills it.
    if sweep_enabled and sweep_center_hold_start_ts > 0:
        g = _ghost_target()
        return g if g is not None else cand
    # NOTE: an earlier "ego lock-down" used to short-circuit to ghost while
    # the chassis was moving. That made pan/tilt blind to the re-emerging
    # bbox after a brief YOLO id drop — visible as "lost the lock while I
    # was driving." Removed. The signature + IoU + hysteresis gates above
    # already keep Pass 1/2/3 from rebinding to a sibling.
    # Count any frame where the resolver wants to rebind, even if YOLO is
    # thrashing the det_id between frames. Without this, hysteresis never
    # confirms when ByteTrack assigns a new id each frame (e.g. after a
    # zoom change) — the camera would drift home while we keep waiting.
    _pending_new_id_streak += 1
    _pending_new_id = cand.det_id
    # Loosen the confirmation count right after a zoom change: ByteTrack
    # drops the ID, every candidate is a "new" one, and a 3-frame wait
    # leaves the camera homing while we hesitate.
    threshold = NEW_ID_CONFIRM_FRAMES
    if zoom_changed_ts > 0 and (time.time() - zoom_changed_ts) < ZOOM_GRACE_S:
        threshold = max(1, NEW_ID_CONFIRM_FRAMES - 2)
    if _pending_new_id_streak >= threshold:
        _pending_new_id = None
        _pending_new_id_streak = 0
        return cand
    g = _ghost_target()
    return g if g is not None else cand


def _bbox_iou(b1, b2) -> float:
    """IoU of two (x1,y1,x2,y2) bboxes. 0 when disjoint or invalid."""
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    a1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _bbox_size_similar(d) -> bool:
    """True if candidate bbox is within BBOX_SIZE_RATIO_TOL of last seen
    bbox size. Stops a small/large nearby detection from being accepted
    as our target when the real target just briefly dropped."""
    w = float(d.bbox[2] - d.bbox[0])
    h = float(d.bbox[3] - d.bbox[1])
    if last_target_w <= 0 or last_target_h <= 0:
        return True
    wr = min(w, last_target_w) / max(w, last_target_w)
    hr = min(h, last_target_h) / max(h, last_target_h)
    return wr >= BBOX_SIZE_RATIO_TOL and hr >= BBOX_SIZE_RATIO_TOL


def _ghost_target() -> Detection | None:
    """Forward-fill a synthetic Detection that simply HOLDS the last known
    bbox center. Extrapolating by velocity feeds back into the controller
    motion and diverges, so we keep the ghost stationary — the controller
    sees zero error and holds position until a real detection returns."""
    if not selected_anchor_xy or last_target_seen_ts <= 0:
        return None
    dt = time.time() - last_target_seen_ts
    if dt > PERSIST_HORIZON_S:
        return None
    ax, ay = selected_anchor_xy
    w2 = last_target_w / 2.0
    h2 = last_target_h / 2.0
    return Detection(
        det_id=selected_id if selected_id is not None else -1,
        name=selected_name or "ghost",
        bbox=(int(ax - w2), int(ay - h2), int(ax + w2), int(ay + h2)),
        score=0.0,
        color=(60, 60, 200),
    )


def resolve_selected_target() -> Detection | None:
    global _resolve_via_same_id, _pending_new_id, _pending_new_id_streak
    _resolve_via_same_id = False
    if selected_name is None and selected_id is None:
        return None
    pool = detections
    if not pool and (time.time() - recent_detection_stamp) < 0.4:
        pool = recent_detections
    strong = [d for d in pool if d.score >= MIN_TRACK_SCORE]
    if not strong:
        strong = list(pool)

    # Advance Kalman prediction to NOW so we score against where the target
    # is expected to be this frame.
    if selected_kf is not None:
        selected_kf.predict()

    # Pre-flight: when ByteTrack still has the locked ID, do a Mahalanobis
    # sanity check + signature check. Only reject the in-frame match if it
    # is wildly inconsistent with the Kalman track AND the appearance has
    # changed — that pattern is YOLO putting our ID on a different physical
    # object (rare but happens).
    if detector_mode in ("auto", "world") and selected_id is not None:
        same_id = next((d for d in strong if d.det_id == selected_id), None)
        if same_id is not None:
            if selected_kf is not None:
                cx = (same_id.bbox[0] + same_id.bbox[2]) / 2
                cy = (same_id.bbox[1] + same_id.bbox[3]) / 2
                m = selected_kf.mahalanobis(cx, cy)
                if m > 200.0:
                    # ID was reassigned to a far-off object. Fall through to
                    # candidate search.
                    pass
                else:
                    _resolve_via_same_id = True
                    _pending_new_id = None
                    _pending_new_id_streak = 0
                    return same_id
            else:
                _resolve_via_same_id = True
                return same_id

    # During the ZOOM_GRACE window right after a zoom change, ByteTrack is
    # likely to have dropped the ID and the bbox geometry is in a new scale,
    # so loosen the gates so the controller doesn't lose the target entirely.
    in_zoom_grace = (zoom_changed_ts > 0 and
                     (time.time() - zoom_changed_ts) < ZOOM_GRACE_S)
    name_gate = (MAX_ANCHOR_REASSOC_PX * 2.5) if in_zoom_grace else MAX_ANCHOR_REASSOC_PX
    cross_gate = (MAX_ANCHOR_REASSOC_CROSS_CLASS_PX * 2.0) if in_zoom_grace else MAX_ANCHOR_REASSOC_CROSS_CLASS_PX

    if detector_mode in ("auto", "world") and selected_id is not None:
        if selected_kf is not None:
            ax, ay = selected_kf.position()
        elif selected_anchor_xy:
            ax, ay = selected_anchor_xy
        else:
            ax, ay = IMG_W / 2.0, IMG_H / 2.0

        if strong:
            # Pass 1: candidates with similar HSV signature, regardless of
            # YOLO class label. Distant cars often get mis-classified as
            # 'truck' or 'airplane' so we can't rely on class alone.
            # Cache sig_d per candidate so we can rank + gap-test.
            sig_pairs = []
            for d in strong:
                sd = _signature_distance(
                    selected_signature,
                    _hsv_signature_from_bbox(latest_raw, d.bbox))
                sig_pairs.append((sd, d))
            sig_pairs.sort(key=lambda p: p[0])
            # Clear-winner gate: when 3+ candidates compete, top must beat
            # 2nd by PASS1_SIG_GAP so near-identical siblings don't get
            # bound interchangeably. With <=2 candidates we skip the gap
            # check (no sibling ambiguity to resolve).
            gap_ok = (len(sig_pairs) < 3) or (sig_pairs[1][0] - sig_pairs[0][0] >= PASS1_SIG_GAP)
            # Predicted bbox at ax,ay scaled to last-seen target size, used
            # for IoU gate so a candidate far from the predicted footprint
            # is rejected even if its histogram matches.
            pred_bbox = None
            if last_target_w > 0 and last_target_h > 0:
                pred_bbox = (ax - last_target_w / 2.0, ay - last_target_h / 2.0,
                             ax + last_target_w / 2.0, ay + last_target_h / 2.0)
            for sig_d, cand in sig_pairs[:3]:
                cx = (cand.bbox[0] + cand.bbox[2]) / 2
                cy = (cand.bbox[1] + cand.bbox[3]) / 2
                m = (selected_kf.mahalanobis(cx, cy)
                     if selected_kf is not None else 0.0)
                iou_ok = True
                if pred_bbox is not None:
                    iou_ok = _bbox_iou(pred_bbox, cand.bbox) >= PASS1_IOU_MIN
                # Wider Kalman + IoU gates while ego is moving — KF
                # process noise can't account for chassis-induced bearing
                # shift, so the prediction drifts away from the real
                # target. Loosen so the re-bind succeeds instead of
                # rejecting on a spurious Maha distance.
                ego_kicked = (mantis_moving
                              or abs(mantis_drive_vx) > 0.1
                              or abs(mantis_drive_vy) > 0.1
                              or abs(mantis_drive_vyaw) > 0.05)
                maha_gate = KF_MAHA_GATE * (4.0 if ego_kicked else 1.0)
                if ego_kicked:
                    iou_ok = True  # rely on signature gate alone
                # Accept only if signature is tight, Kalman gate passes,
                # candidate is a clear winner, and overlaps the predicted
                # footprint.
                if (sig_d < PASS1_SIG_MAX and m <= maha_gate
                        and gap_ok and iou_ok):
                    # Pass 1 winner passed every gate, treat as high-
                    # confidence so the signature can blend toward this
                    # bbox next frame.
                    _resolve_via_same_id = True
                    return _confirm_new_id(cand)
            # Adaptive gate: shrink the spatial radius for small targets so
            # a sibling 60 px away can't be picked when the locked car is
            # only 40 px wide on screen.
            size_gate_limit = max(40.0, last_target_w * 1.5) if last_target_w > 0 else None
            effective_name_gate = (min(name_gate, size_gate_limit)
                                   if size_gate_limit is not None else name_gate)
            effective_cross_gate = (min(cross_gate, size_gate_limit)
                                    if size_gate_limit is not None else cross_gate)
            # Pass 2: same name + within spatial gate + signature gate.
            if selected_name is not None:
                same_name = [d for d in strong if d.name == selected_name]
                if same_name:
                    same_name.sort(
                        key=lambda d: _candidate_score(d, ax, ay, selected_signature)
                    )
                    best = same_name[0]
                    bcx = (best.bbox[0] + best.bbox[2]) / 2
                    bcy = (best.bbox[1] + best.bbox[3]) / 2
                    bsig = _signature_distance(
                        selected_signature,
                        _hsv_signature_from_bbox(latest_raw, best.bbox))
                    if (math.hypot(bcx - ax, bcy - ay) <= effective_name_gate
                            and bsig < 0.55):
                        return _confirm_new_id(best)
            # Pass 3: any class within tight spatial gate + sig gate.
            cross = strong if in_zoom_grace else [d for d in strong if _bbox_size_similar(d)]
            if cross:
                best = _nearest_to_anchor(cross, ax, ay)
                bcx = (best.bbox[0] + best.bbox[2]) / 2
                bcy = (best.bbox[1] + best.bbox[3]) / 2
                bsig = _signature_distance(
                    selected_signature,
                    _hsv_signature_from_bbox(latest_raw, best.bbox))
                if (math.hypot(bcx - ax, bcy - ay) <= effective_cross_gate
                        and bsig < 0.60):
                    return _confirm_new_id(best)
        g = _ghost_target()
        if g is not None:
            return g
        # Persistent memory: ghost horizon expired but selection still alive.
        # Look anywhere in the frame for a same-name detection whose bbox
        # size is similar to the last-seen target — re-bind to it. This is
        # the "the car drove out, came back, please follow it again" path.
        if selected_name is not None and strong:
            dt_lost = (time.time() - last_target_seen_ts) if last_target_seen_ts else 0.0
            if dt_lost <= LONG_LOST_REACQUIRE_S:
                # Same name AND similar bbox size — rejects a near sibling
                # whose footprint is much smaller/larger than the locked
                # target.
                same_name_any = [d for d in strong
                                 if d.name == selected_name and _bbox_size_similar(d)]
                # Score against BOTH the EMA-blended live signature AND the
                # immutable lock-time anchor. Take whichever is closer — if
                # the live signature drifted while we were on a sibling
                # transiently, the anchor catches the original target when
                # it re-emerges from occlusion.
                def _best_sig_dist(d):
                    cand = _hsv_signature_from_bbox(latest_raw, d.bbox)
                    a = _signature_distance(selected_signature, cand) \
                        if selected_signature is not None else 1.0
                    b = _signature_distance(selected_signature_anchor, cand) \
                        if selected_signature_anchor is not None else 1.0
                    return min(a, b)
                if same_name_any and (selected_signature is not None
                                      or selected_signature_anchor is not None):
                    same_name_any.sort(key=_best_sig_dist)
                    best = same_name_any[0]
                    best_sig = _best_sig_dist(best)
                    if best_sig < LONGLOST_SIG_MAX:
                        return _confirm_new_id(best)
        return None

    if selected_name is not None:
        named = [d for d in strong if d.name == selected_name]
    else:
        named = list(strong)
    if not named:
        if selected_anchor_xy and strong:
            ax, ay = selected_anchor_xy
            cross = [d for d in strong if _bbox_size_similar(d)]
            if cross:
                best = _nearest_to_anchor(cross, ax, ay)
                bcx = (best.bbox[0] + best.bbox[2]) / 2
                bcy = (best.bbox[1] + best.bbox[3]) / 2
                if math.hypot(bcx - ax, bcy - ay) <= MAX_ANCHOR_REASSOC_CROSS_CLASS_PX:
                    return best
        return _ghost_target()
    if selected_anchor_xy is None:
        return named[0] if len(named) == 1 else None
    ax, ay = selected_anchor_xy
    sized = [d for d in named if _bbox_size_similar(d)]
    pool = sized if sized else named
    best = _nearest_to_anchor(pool, ax, ay)
    bcx = (best.bbox[0] + best.bbox[2]) / 2
    bcy = (best.bbox[1] + best.bbox[3]) / 2
    dist = math.hypot(bcx - ax, bcy - ay)
    if dist > MAX_ANCHOR_REASSOC_PX:
        return _ghost_target()
    return best


def pixel_norm_to_angle_deg(norm: float, fov_rad: float) -> float:
    return math.degrees(math.atan(math.tan(fov_rad / 2.0) * norm))


def step_toward(target_deg: float, current_deg: float, max_step_deg: float) -> float:
    delta = clamp(target_deg - current_deg, -max_step_deg, max_step_deg)
    return current_deg + delta


def auto_control_step(width: int, height: int, dt: float) -> None:
    global pan_deg, tilt_deg, pan_i_deg, tilt_i_deg, last_ex_norm, last_ey_norm
    global centered_frames, selected_id, selected_name, last_target_ts
    global selected_anchor_xy, sweep_last_advance_ts
    global center_hold_start_ts, sweep_center_hold_start_ts
    global last_target_cx, last_target_cy, last_target_seen_ts
    global target_vx_pix_s, target_vy_pix_s
    global target_world_pan_deg, target_world_tilt_deg
    global target_pan_rate_deg_s, target_tilt_rate_deg_s, last_bearing_ts

    now = time.time()
    target = resolve_selected_target()

    # Sweep watchdog: enforce SWEEP_PER_TARGET_TIMEOUT_S regardless of
    # whether the resolver currently sees the target. Without this, a lost
    # detection traps the sweep in pursuit/ghost mode forever because the
    # sweep state machine below only runs when target is not None.
    if (sweep_enabled and selected_id is not None
            and sweep_last_advance_ts > 0
            and now - sweep_last_advance_ts > SWEEP_PER_TARGET_TIMEOUT_S):
        if selected_id not in sweep_painted_ids:
            sweep_painted_ids.add(selected_id)
        sweep_center_hold_start_ts = 0.0
        clear_selection()
        return

    # Auto-zoom: shrink/expand the digital zoom so the locked target's bbox
    # stays inside a usable size band. Only react to real detections (not
    # ghosts) and respect a cooldown so we don't oscillate.
    global _last_auto_zoom_ts
    if (auto_zoom_enabled and target is not None and target.score > 0
            and selected_id is not None
            and (now - _last_auto_zoom_ts) > AUTO_ZOOM_COOLDOWN_S
            and (now - zoom_changed_ts) > AUTO_ZOOM_COOLDOWN_S):
        tw = float(target.bbox[2] - target.bbox[0])
        th = float(target.bbox[3] - target.bbox[1])
        area_frac = (tw * th) / float(IMG_W * IMG_H)
        new_z = None
        if area_frac < AUTO_ZOOM_LOW and zoom_factor < ZOOM_MAX - 0.05:
            new_z = min(ZOOM_MAX, round((zoom_factor + AUTO_ZOOM_STEP) * 10) / 10)
        elif area_frac > AUTO_ZOOM_HIGH and zoom_factor > ZOOM_MIN + 0.05:
            new_z = max(ZOOM_MIN, round((zoom_factor - AUTO_ZOOM_STEP) * 10) / 10)
        if new_z is not None and abs(new_z - zoom_factor) > 0.05:
            _apply_zoom_locked(new_z)
            _last_auto_zoom_ts = now

    if target is None:
        pan_i_deg *= 0.85
        tilt_i_deg *= 0.85
        last_ex_norm *= 0.5
        last_ey_norm *= 0.5
        target_vx_pix_s *= 0.5
        target_vy_pix_s *= 0.5
        # Active pursuit through detection gaps: extrapolate target bearing
        # using its last-known rate, decay over time. NEVER auto-clear —
        # user keeps authority via the Clear button.
        if (selected_name is not None or selected_id is not None):
            if (target_world_pan_deg is not None
                    and last_bearing_ts > 0):
                dt_lost = now - last_bearing_ts
                if dt_lost <= PURSUIT_MAX_S:
                    decay = math.exp(-dt_lost / PURSUIT_DECAY_S)
                    pred_pan = (target_world_pan_deg
                                + target_pan_rate_deg_s * dt_lost * decay)
                    pred_tilt = (target_world_tilt_deg
                                 + target_tilt_rate_deg_s * dt_lost * decay)
                    pred_pan = clamp(pred_pan, PAN_LIMIT[0], PAN_LIMIT[1])
                    pred_tilt = clamp(pred_tilt, TILT_LIMIT[0], TILT_LIMIT[1])
                    pan_max_step = pan_gains.max_rate_deg_s * dt
                    tilt_max_step = tilt_gains.max_rate_deg_s * dt
                    actual_p = actual_pan_deg if joint_state_stamp else pan_deg
                    actual_t = actual_tilt_deg if joint_state_stamp else tilt_deg
                    pan_step = clamp(pred_pan - pan_deg, -pan_max_step, pan_max_step)
                    tilt_step = clamp(pred_tilt - tilt_deg, -tilt_max_step, tilt_max_step)
                    pan_deg = clamp(pan_deg + 0.35 * pan_step,
                                    PAN_LIMIT[0], PAN_LIMIT[1])
                    tilt_deg = clamp(tilt_deg + 0.35 * tilt_step,
                                     TILT_LIMIT[0], TILT_LIMIT[1])
                    publish_pan_tilt()
                    centered_frames = 0
                    return
            # No motion model yet, or pursuit window expired — freeze.
            publish_pan_tilt()
            centered_frames = 0
            return
        if selected_name is None and selected_id is None:
            # Sweep mode: when idle, pick the first not-yet-painted detection
            # (per det_id — two cars in frame get painted separately).
            if sweep_enabled:
                global sweep_scan_dir, sweep_scan_start_ts
                candidate = next(
                    (d for d in detections
                     if d.score >= MIN_TRACK_SCORE
                     and d.det_id not in sweep_painted_ids),
                    None,
                )
                if candidate is not None:
                    sweep_scan_start_ts = 0.0  # cancel any active scan
                    x1, y1, x2, y2 = candidate.bbox
                    selected_id = candidate.det_id
                    selected_name = candidate.name
                    selected_anchor_xy = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    sweep_last_advance_ts = now
                    return
                # No fresh target in current FOV. Pan the camera to look for
                # one — bouncing between pan limits — instead of resetting
                # painted memory and re-locking the same just-painted target.
                if sweep_scan_start_ts == 0.0:
                    sweep_scan_start_ts = now
                if now - sweep_scan_start_ts > SWEEP_SCAN_MAX_S:
                    sweep_painted_ids.clear()
                    sweep_scan_start_ts = 0.0
                    return
                pan_step = SWEEP_SCAN_RATE_DEG_S * dt * sweep_scan_dir
                new_pan = pan_deg + pan_step
                if new_pan >= PAN_LIMIT[1] - 1.0:
                    new_pan = PAN_LIMIT[1] - 1.0
                    sweep_scan_dir = -1
                elif new_pan <= PAN_LIMIT[0] + 1.0:
                    new_pan = PAN_LIMIT[0] + 1.0
                    sweep_scan_dir = 1
                pan_deg = new_pan
                publish_pan_tilt()
                return
            pan_deg = step_toward(HOME_PAN_DEG, pan_deg, HOME_MAX_RATE_DEG_S * dt)
            tilt_deg = step_toward(HOME_TILT_DEG, tilt_deg, HOME_MAX_RATE_DEG_S * dt)
        else:
            if last_target_ts and now - last_target_ts > LOST_GRACE_S:
                clear_selection()
        pan_deg = clamp(pan_deg, PAN_LIMIT[0], PAN_LIMIT[1])
        tilt_deg = clamp(tilt_deg, TILT_LIMIT[0], TILT_LIMIT[1])
        publish_pan_tilt()
        centered_frames = 0
        return

    last_target_ts = now
    if target.score > 0:
        id_in_frame = any(d.det_id == selected_id for d in detections)
        if selected_id is None or target.det_id == selected_id:
            selected_id = target.det_id
        elif not id_in_frame and target.name == selected_name:
            selected_id = target.det_id
        # Feed measurement to Kalman so the prediction snaps to the
        # observed position. Resolver will use the updated prediction
        # next frame.
        if selected_kf is not None:
            cx_meas = (target.bbox[0] + target.bbox[2]) / 2
            cy_meas = (target.bbox[1] + target.bbox[3]) / 2
            selected_kf.update(cx_meas, cy_meas)
        # Slowly refresh the HSV signature so it tracks gradual
        # appearance change (shadow, viewing angle) without snapping to
        # a sibling on a single bad frame. Only blend on high-confidence
        # frames (resolver returned via the same-id pre-flight branch);
        # blending during Pass-1/ghost/long-lost lets the signature drift
        # toward whatever the resolver guessed and prevents the real
        # target from re-binding.
        global selected_signature
        new_sig = _hsv_signature_from_bbox(latest_raw, target.bbox)
        if new_sig is not None:
            if selected_signature is None:
                selected_signature = new_sig
            elif _resolve_via_same_id:
                selected_signature = 0.85 * selected_signature + 0.15 * new_sig

    x1, y1, x2, y2 = target.bbox
    cx_raw = (x1 + x2) / 2.0
    cy_raw = (y1 + y2) / 2.0
    selected_anchor_xy = (cx_raw, cy_raw)
    if target.score > 0:
        global last_target_w, last_target_h
        new_w = float(x2 - x1)
        new_h = float(y2 - y1)
        # Slew-limit size updates so one wrong-class bbox can't bloat the
        # remembered footprint and unlock the size-similarity gate next
        # frame. High-confidence binds (same-id pre-flight) take the full
        # measurement; uncertain binds blend in slowly.
        if _resolve_via_same_id or last_target_w <= 0 or last_target_h <= 0:
            last_target_w = new_w
            last_target_h = new_h
        else:
            last_target_w = 0.85 * last_target_w + 0.15 * new_w
            last_target_h = 0.85 * last_target_h + 0.15 * new_h

    # Exponential filter on bbox center to remove detector micro-jitter.
    # Higher beta + KF-driven motion between detections so the controller
    # never sits idle waiting for the next YOLO frame.
    global smoothed_cx, smoothed_cy, smoothed_init
    if not smoothed_init:
        smoothed_cx = cx_raw
        smoothed_cy = cy_raw
        smoothed_init = True
    else:
        # 0.20 was too sluggish — caused 50–80 ms latency before pan/tilt
        # saw the new bbox. 0.45 keeps mild jitter rejection but lets the
        # controller respond on the next frame.
        beta = 0.45
        smoothed_cx = (1 - beta) * smoothed_cx + beta * cx_raw
        smoothed_cy = (1 - beta) * smoothed_cy + beta * cy_raw
    cx = smoothed_cx
    cy = smoothed_cy
    # Between detections (camera 30 fps vs YOLO ~12 fps) the smoothed
    # center sits still, so pan/tilt arrives at desired_pan and stalls
    # until the next detection. Project the bbox forward in pixel-space
    # using the EMA velocity so the controller has fresh error to chase
    # every control tick — gives stepper-free continuous slewing.
    if (last_target_seen_ts > 0
            and (now - last_target_seen_ts) < 0.3
            and (abs(target_vx_pix_s) > 5.0 or abs(target_vy_pix_s) > 5.0)):
        stale = now - last_target_seen_ts
        cx = cx + target_vx_pix_s * stale
        cy = cy + target_vy_pix_s * stale

    if last_target_seen_ts > 0:
        gap = max(1e-3, now - last_target_seen_ts)
        if gap < 0.5:
            raw_vx = (cx - last_target_cx) / gap
            raw_vy = (cy - last_target_cy) / gap
            alpha = 0.25
            target_vx_pix_s = (1 - alpha) * target_vx_pix_s + alpha * raw_vx
            target_vy_pix_s = (1 - alpha) * target_vy_pix_s + alpha * raw_vy
        else:
            target_vx_pix_s = 0.0
            target_vy_pix_s = 0.0
    last_target_cx = cx
    last_target_cy = cy
    # Only treat REAL detections as 'seen'. Ghost targets must NOT refresh
    # this stamp or the persistence horizon never expires and the loop
    # keeps fabricating ghosts forever.
    if target.score > 0:
        last_target_seen_ts = now

    # Dead-zone on velocity so detector jitter doesn't drive feedforward.
    vx_eff = target_vx_pix_s if abs(target_vx_pix_s) > 25.0 else 0.0
    vy_eff = target_vy_pix_s if abs(target_vy_pix_s) > 20.0 else 0.0
    lead_s = 0.06
    cx_lead = cx + vx_eff * lead_s
    cy_lead = cy + vy_eff * lead_s

    nx = (cx_lead - width / 2.0) / (width / 2.0)
    ny = (cy_lead - height / 2.0) / (height / 2.0)

    in_deadband_x = abs(nx) < pan_gains.deadband_norm
    in_deadband_y = abs(ny) < tilt_gains.deadband_norm
    ex_eff = 0.0 if in_deadband_x else nx
    ey_eff = 0.0 if in_deadband_y else ny

    pan_err_deg = pixel_norm_to_angle_deg(ex_eff, _eff_hfov)
    tilt_err_deg = pixel_norm_to_angle_deg(ey_eff, _eff_vfov)

    pan_i_deg = clamp(pan_i_deg + pan_err_deg * dt,
                      -pan_gains.integral_clamp_deg, pan_gains.integral_clamp_deg)
    tilt_i_deg = clamp(tilt_i_deg + tilt_err_deg * dt,
                       -tilt_gains.integral_clamp_deg, tilt_gains.integral_clamp_deg)

    # BUG-FIX: previously this used HFOV_RAD/VFOV_RAD (wide-angle constants)
    # while pan_err_deg used _eff_hfov (narrow). At zoom 2x the same nx
    # produced 2x larger pan_err_deg, so the derivative spiked to a
    # gigantic value that whiplashed the controller in the wrong sign.
    last_pan_err_deg = pixel_norm_to_angle_deg(last_ex_norm, _eff_hfov)
    last_tilt_err_deg = pixel_norm_to_angle_deg(last_ey_norm, _eff_vfov)
    # Belt-and-suspenders: clamp the derivative term to a sane range so a
    # one-frame bbox jump can't dominate the loop.
    DERIV_CLAMP_DEG_S = 60.0
    pan_derr_per_s = clamp((pan_err_deg - last_pan_err_deg) / dt,
                           -DERIV_CLAMP_DEG_S, DERIV_CLAMP_DEG_S)
    tilt_derr_per_s = clamp((tilt_err_deg - last_tilt_err_deg) / dt,
                            -DERIV_CLAMP_DEG_S, DERIV_CLAMP_DEG_S)
    last_ex_norm = nx
    last_ey_norm = ny

    # Ego-motion boost: when the chassis is moving, scale Kp + per-frame
    # output clamp so the nose chases the apparent target shift instantly.
    # Default boost = 1.0 (no change). At 1.8 the pan/tilt response on a
    # 4 m/s vehicle catches the target before bbox drifts more than ~5%
    # off center.
    ego_active = (mantis_moving
                  or abs(mantis_drive_vx) > 0.1
                  or abs(mantis_drive_vy) > 0.1
                  or abs(mantis_drive_vyaw) > 0.05)
    ego_boost = 2.6 if ego_active else 1.0
    PID_OUT_CLAMP_DEG = 18.0 if ego_active else 6.0
    pan_u_deg = clamp(
        ego_boost * pan_gains.kp * pan_err_deg
        + pan_gains.ki * pan_i_deg
        + pan_gains.kd * pan_derr_per_s,
        -PID_OUT_CLAMP_DEG, PID_OUT_CLAMP_DEG,
    )
    tilt_u_deg = clamp(
        ego_boost * tilt_gains.kp * tilt_err_deg
        + tilt_gains.ki * tilt_i_deg
        + tilt_gains.kd * tilt_derr_per_s,
        -PID_OUT_CLAMP_DEG, PID_OUT_CLAMP_DEG,
    )

    # Visual servoing: desired joint angle = actual joint angle + correction
    # derived from image error. Ki accumulates against any inner-PID bias so
    # the joint ends up exactly where the image error is zero. No cascaded
    # lead window — that drags cmd around with actual during overshoot.
    joint_fresh = (joint_state_stamp
                   and (time.time() - joint_state_stamp) < JOINT_STALE_S)
    actual_pan = actual_pan_deg if joint_fresh else pan_deg
    actual_tilt = actual_tilt_deg if joint_fresh else tilt_deg

    desired_pan = actual_pan + PAN_SIGN * pan_u_deg
    desired_tilt = actual_tilt + TILT_SIGN * tilt_u_deg

    # Motion-model update: world bearing of the target = actual joint angle
    # plus the angular offset corresponding to its pixel error. Track its
    # rate so pursuit can continue when the bbox briefly vanishes.
    if target.score > 0:
        new_world_pan = actual_pan + PAN_SIGN * pan_err_deg
        new_world_tilt = actual_tilt + TILT_SIGN * tilt_err_deg
        now_bts = time.time()
        if target_world_pan_deg is not None and last_bearing_ts > 0:
            dt_b = max(1e-3, now_bts - last_bearing_ts)
            raw_pan_rate = (new_world_pan - target_world_pan_deg) / dt_b
            raw_tilt_rate = (new_world_tilt - target_world_tilt_deg) / dt_b
            beta = 0.30
            target_pan_rate_deg_s = (1 - beta) * target_pan_rate_deg_s + beta * raw_pan_rate
            target_tilt_rate_deg_s = (1 - beta) * target_tilt_rate_deg_s + beta * raw_tilt_rate
        target_world_pan_deg = new_world_pan
        target_world_tilt_deg = new_world_tilt
        last_bearing_ts = now_bts

    # Rate limit + low-pass filter on outgoing command for smooth motion.
    # At higher zoom the effective FOV shrinks. A normal max_rate would slew
    # the camera past the visible window before the next detection arrives,
    # so we scale the rate cap inversely with zoom.
    # Aggressive rate scaling: pixel-per-degree grows with zoom, so any
    # over-correction looks proportionally larger. Scale max slew by 1/z^1.4.
    zoom_scale = 1.0 / (max(1.0, zoom_factor) ** 1.4)
    # Ego boost also widens the per-step rate cap + sharpens the LPF so the
    # joint can actually deliver the higher PID output. Joint hardware
    # limits are pan 2.5 rad/s (143 deg/s), tilt 2.0 rad/s (114 deg/s) —
    # well above the boosted gains so the actuator isn't the bottleneck.
    rate_boost = 2.5 if ego_active else 1.0
    pan_max_step = pan_gains.max_rate_deg_s * dt * zoom_scale * rate_boost
    tilt_max_step = tilt_gains.max_rate_deg_s * dt * zoom_scale * rate_boost
    # Feed-forward: chassis yaw rotates the camera in world frame. Subtract
    # it from the pan command directly so the nose stays glued to the
    # world-frame bearing without waiting for the feedback loop to detect
    # the bbox shift. dyaw is in rad/s; convert to deg/s and to per-frame
    # step. Sign flipped to match camera convention.
    yaw_ff_deg = -math.degrees(mantis_drive_vyaw) * dt
    desired_pan_ff = desired_pan + yaw_ff_deg
    pan_step_raw = clamp(desired_pan_ff - pan_deg, -pan_max_step, pan_max_step)
    tilt_step_raw = clamp(desired_tilt - tilt_deg, -tilt_max_step, tilt_max_step)
    base_lpf = 0.32 / max(1.0, math.sqrt(zoom_factor))
    lpf = min(0.7, base_lpf * 1.8) if ego_active else base_lpf

    # Inside deadband: freeze command at actual joint position and decay
    # integral fast so we don't accumulate noise into the next motion.
    if in_deadband_x:
        pan_deg = clamp(0.85 * pan_deg + 0.15 * actual_pan,
                        PAN_LIMIT[0], PAN_LIMIT[1])
        pan_i_deg *= 0.80
    else:
        pan_deg = clamp(pan_deg + lpf * pan_step_raw,
                        PAN_LIMIT[0], PAN_LIMIT[1])
    if in_deadband_y:
        tilt_deg = clamp(0.85 * tilt_deg + 0.15 * actual_tilt,
                         TILT_LIMIT[0], TILT_LIMIT[1])
        tilt_i_deg *= 0.80
    else:
        tilt_deg = clamp(tilt_deg + lpf * tilt_step_raw,
                         TILT_LIMIT[0], TILT_LIMIT[1])
    publish_pan_tilt()

    if abs(nx) < pan_gains.deadband_norm and abs(ny) < tilt_gains.deadband_norm:
        centered_frames += 1
        if center_hold_start_ts == 0.0:
            center_hold_start_ts = now
    else:
        centered_frames = 0
        center_hold_start_ts = 0.0
    # Sweep painter uses a far looser centering tolerance — bbox jitter at
    # 30 m makes the tight controller deadband never trigger.
    if abs(nx) < SWEEP_CENTER_TOL_NORM and abs(ny) < SWEEP_CENTER_TOL_NORM:
        if sweep_center_hold_start_ts == 0.0:
            sweep_center_hold_start_ts = now
    else:
        sweep_center_hold_start_ts = 0.0
    if (paint_auto and centered_frames == paint_auto_min_centered
            and time.time() - paint_last_ts > 0.4):
        trigger_paint("auto-center-hold")

    if sweep_enabled:
        # Auto-serial state machine: hold center for SWEEP_PAINT_HOLD_S,
        # paint THIS instance, mark its det_id painted, advance to next.
        # If the target was manually clicked (not picked by sweep itself)
        # sweep_last_advance_ts is 0 — initialize so the timeout below can
        # still fire and keep us moving forward.
        if sweep_last_advance_ts == 0.0:
            sweep_last_advance_ts = now
        held_s = ((now - sweep_center_hold_start_ts)
                  if sweep_center_hold_start_ts else 0.0)
        already_painted = selected_id is not None and selected_id in sweep_painted_ids
        if (held_s >= SWEEP_PAINT_HOLD_S
                and now - paint_last_ts > 0.3
                and selected_id is not None
                and not already_painted):
            trigger_paint(f"sweep:{target.name}#{selected_id}")
            sweep_painted_ids.add(selected_id)
            sweep_center_hold_start_ts = 0.0
            clear_selection()
            return
        # If the resolver re-bound onto an id we've already painted (common
        # after a YOLO id flip), skip immediately rather than waiting on the
        # timeout — otherwise we'd burn 5 s of "centering" on a done target.
        if already_painted:
            clear_selection()
            return
        if now - sweep_last_advance_ts > SWEEP_PER_TARGET_TIMEOUT_S:
            # Couldn't center in time. Mark this instance done so we move on.
            if selected_id is not None:
                sweep_painted_ids.add(selected_id)
            clear_selection()
            return


def manual_control_step(dt: float) -> None:
    global pan_deg, tilt_deg, jog_pan_target, jog_tilt_target
    if jog_pan_target is not None:
        pan_deg = step_toward(jog_pan_target, pan_deg,
                              pan_gains.max_rate_deg_s * dt)
        if abs(jog_pan_target - pan_deg) < 0.05:
            jog_pan_target = None
    if jog_tilt_target is not None:
        tilt_deg = step_toward(jog_tilt_target, tilt_deg,
                               tilt_gains.max_rate_deg_s * dt)
        if abs(jog_tilt_target - tilt_deg) < 0.05:
            jog_tilt_target = None
    pan_deg = clamp(pan_deg, PAN_LIMIT[0], PAN_LIMIT[1])
    tilt_deg = clamp(tilt_deg, TILT_LIMIT[0], TILT_LIMIT[1])
    publish_pan_tilt()


def home_control_step(dt: float) -> None:
    global pan_deg, tilt_deg
    pan_deg = step_toward(HOME_PAN_DEG, pan_deg, HOME_MAX_RATE_DEG_S * dt)
    tilt_deg = step_toward(HOME_TILT_DEG, tilt_deg, HOME_MAX_RATE_DEG_S * dt)
    pan_deg = clamp(pan_deg, PAN_LIMIT[0], PAN_LIMIT[1])
    tilt_deg = clamp(tilt_deg, TILT_LIMIT[0], TILT_LIMIT[1])
    publish_pan_tilt()


def stop_control_step() -> None:
    global last_command_pan_deg, last_command_tilt_deg
    publish_angle(pan_pub, last_command_pan_deg)
    publish_angle(tilt_pub, last_command_tilt_deg)


CAMERA_STALE_S = 1.5
JOINT_STALE_S = 2.0


def control_tick(width: int, height: int) -> None:
    global last_control_ts, last_command_pan_deg, last_command_tilt_deg
    now = time.time()
    dt = (now - last_control_ts) if last_control_ts else 1.0 / 30.0
    dt = clamp(dt, 1e-3, 0.2)
    last_control_ts = now

    if mode == "stop":
        stop_control_step()
        return
    if mode == "passthrough":
        # autotune / external publisher owns /mantis/pan_cmd & /mantis/tilt_cmd
        return
    # Watchdog: if camera frames stop, freeze command rather than running
    # blind. Detection list is also stale, so chasing it is dangerous.
    if latest_stamp and (now - latest_stamp) > CAMERA_STALE_S:
        stop_control_step()
        return

    # Sweep is independent of mode: when ON it always runs the autonomous
    # pick → center → paint → next loop, even if Tracking is OFF.
    if sweep_enabled:
        auto_control_step(width, height, dt)
    elif mode == "auto":
        auto_control_step(width, height, dt)
    elif mode == "manual":
        manual_control_step(dt)
    else:
        home_control_step(dt)
    last_command_pan_deg = pan_deg
    last_command_tilt_deg = tilt_deg


def draw_overlay(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.line(out, (w // 2 - 36, h // 2), (w // 2 + 36, h // 2), (255, 255, 255), 1)
    cv2.line(out, (w // 2, h // 2 - 36), (w // 2, h // 2 + 36), (255, 255, 255), 1)
    cv2.rectangle(out, (w // 2 - 46, h // 2 - 32), (w // 2 + 46, h // 2 + 32),
                  (80, 220, 255), 1)
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        is_sel = (det.det_id == selected_id)
        if is_sel:
            color = (0, 0, 255)  # red (BGR) — the actively tracked target
            thickness = 4
        else:
            color = det.color
            thickness = 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"ID {det.det_id} {det.name}"
        if is_sel:
            label = "[LOCKED] " + label
        cv2.putText(out, label, (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    now = time.time()
    for m in list(paint_overlay_marks):
        age = now - m["ts"]
        if age > PAINT_OVERLAY_TTL_S:
            paint_overlay_marks.remove(m)
            continue
        frac = age / PAINT_OVERLAY_TTL_S
        radius = int(18 + 38 * frac)
        thickness = max(2, int(6 * (1.0 - frac)))
        col = m["color"]
        cv2.circle(out, (int(m["cx"]), int(m["cy"])), radius, col, thickness)
        cv2.circle(out, (int(m["cx"]), int(m["cy"])), max(3, radius // 4), col, -1)
        cv2.putText(out, "PAINT", (int(m["cx"]) - 30, int(m["cy"]) - radius - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

    cv2.rectangle(out, (12, 12), (640, 132), (10, 12, 14), -1)
    cv2.putText(out, f"MANTIS NOSE CAMERA  [{mode.upper()}]",
                (28, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 245, 250), 2)
    cv2.putText(out,
                f"pan {pan_deg:6.1f} deg  tilt {tilt_deg:5.1f} deg  sel {selected_name or '-'}",
                (28, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (240, 245, 250), 2)
    det_label = "YOLOv12+ByteTrack" if detector_mode == "auto" else "HSV+AnchorMatch"
    cv2.putText(out,
                f"{det_label}  paint:{paint_count}  Kp {pan_gains.kp:.2f} Ki {pan_gains.ki:.2f} Kd {pan_gains.kd:.2f}",
                (28, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 200, 220), 2)
    return out


_detection_thread_busy = False
_detection_busy_since = 0.0
_detection_last_dispatch = 0.0
_detection_last_completed = 0.0
_detection_last_latency_s = 0.0
DETECTION_INTERVAL_S = 1.0 / 12.0  # 12 Hz — between flicker resilience and CPU
DETECTION_BUSY_TIMEOUT_S = 3.0    # force-reset busy flag if worker hangs


def run_detector(frame: np.ndarray) -> list[Detection]:
    if detector_mode == "color":
        return detect_colored_targets(frame)
    if detector_mode == "world":
        wd = detect_with_yoloworld(frame)
        if wd is not None:
            return wd
        # fall through to closed-set YOLO if world load fails
    yolo_dets = detect_with_yolo(frame)
    if yolo_dets is None:
        return detect_colored_targets(frame)
    return yolo_dets


def detection_worker(frame: np.ndarray) -> None:
    global detections, recent_detections, recent_detection_stamp
    global _detection_thread_busy, yolo_status
    global _detection_last_completed, _detection_last_latency_s
    t0 = time.time()
    try:
        try:
            result = run_detector(frame)
        except Exception as exc:
            yolo_status = f"detector exception: {exc}"
            result = []
        with lock:
            detections = result
            if result:
                recent_detections = list(result)
                recent_detection_stamp = time.time()
            _detection_last_completed = time.time()
            _detection_last_latency_s = _detection_last_completed - t0
    finally:
        _detection_thread_busy = False


def _process_raw_frame(raw: np.ndarray) -> None:
    """Shared pipeline: zoom → publish to UI → detector → controller. Used
    by both the gz subscriber callback and any custom capture thread
    (webcam, RTSP, ...)."""
    global latest_raw, latest_annotated, latest_stamp, frame_count
    global _detection_thread_busy, _eff_hfov, _eff_vfov
    frame, eff_h, eff_v = apply_digital_zoom(raw)
    with lock:
        latest_raw = frame
        _eff_hfov = eff_h
        _eff_vfov = eff_v
        control_tick(frame.shape[1], frame.shape[0])
        annotated = draw_overlay(frame)
        latest_stamp = time.time()
        frame_count += 1
    ok, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 72])
    if ok:
        latest_annotated = jpg.tobytes()
    global _detection_last_dispatch, _detection_thread_busy, _detection_busy_since
    now = time.time()
    # Force-reset busy flag if a previous worker has been stuck too long
    # (e.g. YOLO inference hung, model reload, or exception escaped lock).
    if _detection_thread_busy and (now - _detection_busy_since) > DETECTION_BUSY_TIMEOUT_S:
        _detection_thread_busy = False
    if (not _detection_thread_busy
            and (now - _detection_last_dispatch) >= DETECTION_INTERVAL_S):
        _detection_thread_busy = True
        _detection_busy_since = now
        _detection_last_dispatch = now
        threading.Thread(target=detection_worker, args=(frame,),
                         daemon=True).start()


def on_image(msg: Image) -> None:
    """gz transport callback. Decodes the gz Image and feeds the shared
    pipeline."""
    raw = image_to_bgr(msg)
    if raw is None:
        return
    _process_raw_frame(raw)


# --- Pluggable frame sources -------------------------------------------------
# UI's Connect dropdown switches between gz (Gazebo sim) and webcam capture.
# All sources hand a BGR ndarray to _process_raw_frame; the rest of the
# pipeline is unchanged.
frame_source = "gz"                  # "gz" | "webcam:<idx>"
_webcam_thread: threading.Thread | None = None
_webcam_stop = threading.Event()
_webcam_cap = None
_webcam_last_error = ""


def _webcam_loop(index: int) -> None:
    global _webcam_cap, _webcam_last_error
    try:
        _webcam_cap = cv2.VideoCapture(index)
        # ask for 1280x720 — if camera can't, opencv falls back gracefully
        _webcam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
        _webcam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)
        _webcam_cap.set(cv2.CAP_PROP_FPS, 30)
        if not _webcam_cap.isOpened():
            _webcam_last_error = f"could not open webcam index {index}"
            return
    except Exception as exc:
        _webcam_last_error = f"webcam open exception: {exc}"
        return
    _webcam_last_error = ""
    while not _webcam_stop.is_set():
        ok, frame = _webcam_cap.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue
        # Resize to canonical frame size so the rest of the pipeline (anchor
        # gates, deadbands) works without retuning.
        if frame.shape[1] != IMG_W or frame.shape[0] != IMG_H:
            frame = cv2.resize(frame, (IMG_W, IMG_H))
        _process_raw_frame(frame)
    try:
        _webcam_cap.release()
    except Exception:
        pass
    _webcam_cap = None


def _stop_webcam() -> None:
    global _webcam_thread
    _webcam_stop.set()
    if _webcam_thread is not None and _webcam_thread.is_alive():
        _webcam_thread.join(timeout=2.0)
    _webcam_thread = None
    _webcam_stop.clear()


def _start_webcam(index: int) -> None:
    global _webcam_thread
    _stop_webcam()
    _webcam_thread = threading.Thread(target=_webcam_loop, args=(index,),
                                      daemon=True)
    _webcam_thread.start()


def on_joint_state(msg: Model) -> None:
    global actual_pan_deg, actual_tilt_deg, pan_vel_deg_s, tilt_vel_deg_s
    global joint_state_stamp
    for j in msg.joint:
        if j.name == "pan_joint":
            actual_pan_deg = math.degrees(j.axis1.position)
            pan_vel_deg_s = math.degrees(j.axis1.velocity)
        elif j.name == "tilt_joint":
            actual_tilt_deg = math.degrees(j.axis1.position)
            tilt_vel_deg_s = math.degrees(j.axis1.velocity)
    joint_state_stamp = time.time()


target_moving = False
cars_moving = False
people_walking = False
bus_moving = False
mantis_moving = False
MANTIS_SWAY_AMP = 4.0    # meters: peak +/- displacement from spawn
MANTIS_SWAY_PERIOD_S = 6.0
# Live mantis chassis pose for V-key drive. Integrated continuously by
# _mantis_drive_loop using mantis_drive_v* (m/s, rad/s).
mantis_chassis_x: float = 0.0
mantis_chassis_y: float = -8.0
mantis_chassis_yaw: float = 0.0
mantis_drive_vx: float = 0.0
mantis_drive_vy: float = 0.0
mantis_drive_vyaw: float = 0.0
MANTIS_DRIVE_MAX_SPEED = 25.0  # m/s clamp (high-speed mode caps)
MANTIS_DRIVE_MAX_YAW = 3.0     # rad/s clamp
mantis_drive_speed: float = 4.0  # active speed setpoint used by V-key drive
BUS_SPEED_M_S = 4.0       # peak depth velocity (front-back from mantis perspective)
BUS_PERIOD_S = 3.0        # full forward+reverse cycle


def _bus_loop():
    """Drives ring_bus_a back and forth by teleporting via the gz set_pose
    service. VelocityControl was flaky on this rigid inline model — pose
    teleport guarantees clean depth motion at any speed."""
    import math as _m
    import subprocess as _sp
    # Bus spawn at (18, 6, 1.6) yaw=0.665 rad. Body-x direction in world is
    # (cos(yaw), sin(yaw)) = (0.787, 0.617). Oscillate along that vector.
    BX, BY, BZ, YAW = 0.0, 42.0, 1.6, 1.5708
    # Body-x at yaw=pi/2 points along world +y → depth axis from mantis at (0,-8).
    dirx, diry = _m.cos(YAW), _m.sin(YAW)
    t0 = time.time()
    while True:
        if bus_moving:
            # Triangle wave: amplitude = BUS_SPEED * PERIOD/4. Linear back+forth.
            phase = ((time.time() - t0) % BUS_PERIOD_S) / BUS_PERIOD_S  # 0..1
            if phase < 0.5:
                travel = (phase * 4 - 1) * BUS_SPEED_M_S * BUS_PERIOD_S / 4
            else:
                travel = (3 - phase * 4) * BUS_SPEED_M_S * BUS_PERIOD_S / 4
            x = BX + dirx * travel
            y = BY + diry * travel
            req = (f'name: "ring_bus_a" '
                   f'position {{ x: {x:.3f} y: {y:.3f} z: {BZ:.3f} }} '
                   f'orientation {{ z: {_m.sin(YAW/2):.4f} w: {_m.cos(YAW/2):.4f} }}')
            try:
                _sp.run(
                    ["gz", "service", "-s", "/world/mantis_robot_world/set_pose",
                     "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
                     "--timeout", "200", "--req", req],
                    capture_output=True, timeout=1.0, check=False,
                )
            except Exception:
                pass
        time.sleep(0.05)


threading.Thread(target=_bus_loop, daemon=True).start()


def _mantis_motion_loop():
    """Ego-motion: teleport the mantis chassis along world +x to simulate a
    moving turret (vehicle, gimbal on a rover, ...). The tracker has to
    cope with both target motion AND camera-frame motion."""
    import math as _m, subprocess as _sp
    SPAWN_X, SPAWN_Y, SPAWN_Z = 0.0, -8.0, 0.0
    t0 = time.time()
    while True:
        if mantis_moving:
            phase = ((time.time() - t0) % MANTIS_SWAY_PERIOD_S) / MANTIS_SWAY_PERIOD_S
            # triangle wave on +/- amplitude
            if phase < 0.5:
                travel = (phase * 4 - 1) * MANTIS_SWAY_AMP
            else:
                travel = (3 - phase * 4) * MANTIS_SWAY_AMP
            x = SPAWN_X + travel
            req = (f'name: "mantis_robot" '
                   f'position {{ x: {x:.3f} y: {SPAWN_Y:.3f} z: {SPAWN_Z:.3f} }} '
                   f'orientation {{ w: 1.0 }}')
            try:
                _sp.run(
                    ["gz", "service", "-s", "/world/mantis_robot_world/set_pose",
                     "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
                     "--timeout", "200", "--req", req],
                    capture_output=True, timeout=1.0, check=False,
                )
            except Exception:
                pass
        time.sleep(0.05)


threading.Thread(target=_mantis_motion_loop, daemon=True).start()


def _mantis_drive_loop():
    """30 Hz integrator that turns velocity setpoints into smooth chassis
    motion. JS keydown writes mantis_drive_v*; keyup zeros them; this loop
    advances pose and teleports via gz set_pose. Gives car-like continuous
    motion instead of per-keypress jumps."""
    import math as _m, subprocess as _sp
    global mantis_chassis_x, mantis_chassis_y, mantis_chassis_yaw
    dt = 1.0 / 30.0
    while True:
        if abs(mantis_drive_vx) > 1e-3 or abs(mantis_drive_vy) > 1e-3 \
                or abs(mantis_drive_vyaw) > 1e-3:
            # vx is FORWARD (body x), vy is LEFT (body y). Rotate into world.
            yaw = mantis_chassis_yaw
            wx = mantis_drive_vx * _m.cos(yaw) - mantis_drive_vy * _m.sin(yaw)
            wy = mantis_drive_vx * _m.sin(yaw) + mantis_drive_vy * _m.cos(yaw)
            mantis_chassis_x += wx * dt
            mantis_chassis_y += wy * dt
            mantis_chassis_yaw += mantis_drive_vyaw * dt
            req = (f'name: "mantis_robot" '
                   f'position {{ x: {mantis_chassis_x:.3f} y: {mantis_chassis_y:.3f} z: 0.0 }} '
                   f'orientation {{ z: {_m.sin(mantis_chassis_yaw/2):.4f} '
                   f'w: {_m.cos(mantis_chassis_yaw/2):.4f} }}')
            try:
                _sp.run(
                    ["gz", "service", "-s", "/world/mantis_robot_world/set_pose",
                     "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
                     "--timeout", "100", "--req", req],
                    capture_output=True, timeout=0.3, check=False,
                )
            except Exception:
                pass
        time.sleep(dt)


threading.Thread(target=_mantis_drive_loop, daemon=True).start()


def _moving_target_loop():
    """Yellow sphere bounces in place. Prius + pickup drive in a true
    smooth circle in front of the mantis using VelocityControl on each
    model's cmd_vel topic (so the motion is continuous in the physics
    simulator, not teleported every 200 ms via set_pose).

    Each car gets:
        linear.x  =  speed   (forward in body frame)
        angular.z =  speed / radius  (yaw rate that draws a circle)
    """
    import math as _m
    CIRCLE_R = 10.0
    SPEED = 1.6                 # m/s tangential — slow enough for the camera to follow
    OMEGA = SPEED / CIRCLE_R
    t0 = time.time()
    while True:
        t = time.time() - t0
        # Yellow bouncing ball
        ball = Twist()
        if target_moving:
            ball.linear.z = 2.5 * _m.sin(2.0 * t)
        try:
            moving_target_pub.publish(ball)
        except Exception:
            pass
        # Prius and pickup drive their own circles
        prius = Twist()
        pickup = Twist()
        if cars_moving:
            prius.linear.x = SPEED
            prius.angular.z = OMEGA
            pickup.linear.x = SPEED
            pickup.angular.z = -OMEGA  # opposite turn direction
        try:
            car_prius_pub.publish(prius)
            car_pickup_pub.publish(pickup)
        except Exception:
            pass
        # Two extra colored spheres bounce in place at distinct frequencies.
        # Replaces the standing-person models (which fall over under
        # VelocityControl). YOLO detects these reliably as 'sports ball'
        # so they give us a second tracking class to test against.
        r = Twist()
        g = Twist()
        if people_walking:
            r.linear.z = 2.0 * _m.sin(1.7 * t)
            g.linear.z = 2.2 * _m.sin(2.3 * t + 1.0)
        try:
            ball_red_pub.publish(r)
            ball_green_pub.publish(g)
        except Exception:
            pass
        time.sleep(0.04)


threading.Thread(target=_moving_target_loop, daemon=True).start()


def camera_thread() -> None:
    node.subscribe(Image, CAMERA_TOPIC, on_image)
    node.subscribe(Model, JOINT_STATE_TOPIC, on_joint_state)
    while True:
        time.sleep(1.0)


threading.Thread(target=camera_thread, daemon=True).start()


def _yolo_prewarm():
    """Load YOLO + run one dummy inference so the first real frame
    doesn't pay the cold-start hit (which was the cause of 'no detections
    until I click Manual' — the first dispatch was eating ~1.5 s loading
    the model and another ~0.5 s warming the torch graph)."""
    m = _load_yolo()
    if m is None:
        return
    try:
        dummy = np.zeros((384, 640, 3), dtype=np.uint8)
        m.predict(source=dummy, imgsz=384, conf=0.30, iou=0.5, verbose=False)
    except Exception as exc:
        global yolo_status
        yolo_status = f"prewarm warn: {exc}"


threading.Thread(target=_yolo_prewarm, daemon=True).start()


HTML_PAGE = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MANTIS PAINTER Tracker</title>
  <style>
    :root{color-scheme:dark;--bg:#101214;--panel:#181b1f;--line:#2b3138;--text:#eef2f5;--muted:#9da7b1;--cyan:#56cfe1;--amber:#ffd166;--green:#5bd97f;--red:#ff6b6b}
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,Segoe UI,sans-serif}
    header{height:54px;padding:0 18px;display:flex;align-items:center;justify-content:space-between;background:#0b0d10;border-bottom:1px solid var(--line)}
    h1{font-size:18px;margin:0}
    main{display:grid;grid-template-columns:minmax(640px,1.35fr) minmax(380px,.65fr);gap:12px;padding:12px}
    section{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}
    .title{height:38px;display:flex;align-items:center;justify-content:space-between;padding:0 12px;border-bottom:1px solid var(--line);color:var(--muted);font-size:13px}
    #feedWrap{position:relative;overflow:hidden;background:#050607;aspect-ratio:16/9;cursor:crosshair}
    #feed{width:100%;aspect-ratio:16/9;display:block;background:#050607;cursor:crosshair}
    .shortcuts{position:absolute;top:8px;right:8px;background:rgba(8,11,14,.72);border:1px solid #2b3138;border-radius:6px;padding:6px 9px;font-size:11px;line-height:1.45;color:#cfd6dc;pointer-events:none;font-family:ui-monospace,Menlo,Consolas,monospace;backdrop-filter:blur(2px)}
    .shortcuts b{color:var(--cyan);font-weight:600}
    .shortcuts .hd{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:3px}
    .stats{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;padding:10px}
    .stat{border:1px solid var(--line);border-radius:6px;padding:10px;min-height:62px}
    .label{color:var(--muted);font-size:12px}
    .value{font-size:20px;margin-top:6px;font-variant-numeric:tabular-nums}
    button{height:32px;background:#26313a;color:var(--text);border:1px solid #39434e;border-radius:6px;padding:0 10px;cursor:pointer}
    button:hover{background:#303c47}
    button.active{background:var(--cyan);color:#06181c;border-color:var(--cyan)}
    .row{display:flex;gap:6px;padding:10px;flex-wrap:wrap}
    .pad{display:grid;grid-template-columns:repeat(3,44px);grid-template-rows:repeat(3,40px);gap:6px;justify-content:center;padding:10px}
    .pad button{height:40px;width:44px;padding:0;font-size:18px}
    .pad .sp{visibility:hidden}
    .gain{display:grid;grid-template-columns:48px 1fr 56px;gap:8px;align-items:center;padding:4px 12px;font-size:13px}
    .gain input[type=range]{width:100%}
    .text{padding:12px;color:#d5dbe1;line-height:1.45;font-size:14px}
    table{width:100%;border-collapse:collapse;font-size:13px}
    td,th{padding:6px 10px;border-top:1px solid var(--line);text-align:left}
    .sect-head{padding:8px 12px 0;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}
    select{height:28px;background:#26313a;color:var(--text);border:1px solid #39434e;border-radius:6px;padding:0 6px}
  </style>
</head>
<body>
<header>
  <div style="display:flex;align-items:center;gap:12px">
    <h1>MANTIS PAINTER &mdash; Nose Camera Tracking</h1>
    <div style="position:relative">
      <button id="bConnect" title="Connection settings: choose video source + configure paint signal output (gz/file/UDP/TCP/serial).">Connect: Gazebo Sim</button>
      <div id="connectMenu" style="display:none;position:absolute;top:38px;left:0;background:#181b1f;border:1px solid var(--line);border-radius:8px;padding:10px;z-index:1000;width:420px;box-shadow:0 8px 24px rgba(0,0,0,.4)">
        <div class="sect-head" style="padding:0 0 6px">Frame source</div>
        <button class="connOpt" data-src="auto" style="display:block;width:100%;text-align:left;height:30px;margin-bottom:6px;background:#1f4d8c;border-color:#3a78c0;color:#fff" title="Probe Gazebo (camera topic + joint state) and webcams 0/1/2, lock onto the first one that responds.">⚡ Auto Detect &amp; Connect</button>
        <button class="connOpt" data-src="gz" style="display:block;width:100%;text-align:left;height:30px;margin-bottom:4px">Gazebo Sim (/mantis/nose_camera/image)</button>
        <button class="connOpt" data-src="webcam:0" style="display:block;width:100%;text-align:left;height:30px;margin-bottom:4px">Laptop Webcam #0</button>
        <button class="connOpt" data-src="webcam:1" style="display:block;width:100%;text-align:left;height:30px;margin-bottom:4px">USB Webcam #1</button>
        <button class="connOpt" data-src="webcam:2" style="display:block;width:100%;text-align:left;height:30px;margin-bottom:8px">USB Webcam #2</button>
        <div class="sect-head" style="padding:6px 0">Paint signal channels</div>
        <div style="display:flex;flex-wrap:wrap;gap:8px;padding-bottom:6px">
          <label class="label"><input type="checkbox" id="chGz" checked> gz topic</label>
          <label class="label"><input type="checkbox" id="chFile" checked> file</label>
          <label class="label"><input type="checkbox" id="chUdp"> UDP</label>
          <label class="label"><input type="checkbox" id="chTcp"> TCP</label>
          <label class="label"><input type="checkbox" id="chSerial"> serial</label>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px 10px;padding-bottom:4px">
          <label class="label">UDP host <input id="udpHost" value="127.0.0.1" style="width:100%;background:#1d2227;color:var(--text);border:1px solid #39434e;border-radius:4px;padding:2px 6px;margin-top:2px"></label>
          <label class="label">UDP port <input id="udpPort" value="9000" style="width:100%;background:#1d2227;color:var(--text);border:1px solid #39434e;border-radius:4px;padding:2px 6px;margin-top:2px"></label>
          <label class="label">TCP host <input id="tcpHost" value="127.0.0.1" style="width:100%;background:#1d2227;color:var(--text);border:1px solid #39434e;border-radius:4px;padding:2px 6px;margin-top:2px"></label>
          <label class="label">TCP port <input id="tcpPort" value="9001" style="width:100%;background:#1d2227;color:var(--text);border:1px solid #39434e;border-radius:4px;padding:2px 6px;margin-top:2px"></label>
          <label class="label">Serial port <input id="serPort" value="/dev/ttyUSB0" style="width:100%;background:#1d2227;color:var(--text);border:1px solid #39434e;border-radius:4px;padding:2px 6px;margin-top:2px"></label>
          <label class="label">Baud <input id="serBaud" value="9600" style="width:100%;background:#1d2227;color:var(--text);border:1px solid #39434e;border-radius:4px;padding:2px 6px;margin-top:2px"></label>
        </div>
        <div style="display:flex;justify-content:flex-end;gap:6px;padding-top:6px">
          <button id="chSave">Apply channels</button>
        </div>
      </div>
    </div>
  </div>
  <div class="label">Click target. Pan/tilt centers it. Manual mode + jog enabled.</div>
</header>
<main>
  <div>
    <section>
      <div class="title">
        <span>Live Nose Camera</span>
        <span style="display:flex;gap:8px;align-items:center">
          <label class="label">zoom <input id="zoom" type="range" min="1" max="4" step="0.1" value="1" style="width:120px;vertical-align:middle"></label>
          <span id="zoomV" class="label">1.0x</span>
          <span id="status">connecting</span>
        </span>
      </div>
      <div id="feedWrap">
        <img id="feed" src="/video_feed">
        <div class="shortcuts">
          <div class="hd">Shortcuts</div>
          <div><b>L</b> lock center · <b>C</b> clear · <b>A</b> auto · <b>S</b> stop · <b>H</b> home</div>
          <div><b>K</b> keyboard on/off · <b>Shift+&uarr;/&darr;</b> zoom · <b>Esc</b> stop · <b>P</b>/Space paint</div>
          <div><b>M</b> drive nose (pan/tilt) · <b>V</b> drive chassis (front/back/left/right)</div>
          <div><b>Arrows</b> jog (Keyboard ON) · click image to select · <b>T</b> tracking</div>
        </div>
      </div>
    </section>
    <section style="margin-top:12px">
      <div class="title">
        <span>Agent (Ollama)</span>
        <span style="display:flex;gap:6px;align-items:center">
          <select id="agentModel" style="max-width:160px"></select>
          <button id="bAgentToggle">Agent: OFF</button>
        </span>
      </div>
      <div id="chatLog" style="height:180px;overflow-y:auto;padding:8px 12px;font-size:13px;background:#0d1013;border-bottom:1px solid var(--line)"></div>
      <div class="row" style="padding:8px 10px">
        <input id="chatIn" type="text" placeholder="tell the agent (e.g. paint the car)" style="flex:1;height:30px;background:#1d2227;color:var(--text);border:1px solid #39434e;border-radius:6px;padding:0 8px">
        <button id="chatSend">Send</button>
      </div>
    </section>
  </div>
  <div>
    <section>
      <div class="title"><span>Controller</span><span id="modeBadge">auto</span></div>
      <div class="stats">
        <div class="stat"><div class="label">Selected</div><div id="selected" class="value">none</div></div>
        <div class="stat"><div class="label">Detections</div><div id="detCount" class="value">0</div></div>
        <div class="stat"><div class="label">Pan&deg;</div><div id="pan" class="value">0.0</div></div>
        <div class="stat"><div class="label">Tilt&deg;</div><div id="tilt" class="value">0.0</div></div>
      </div>

      <div class="sect-head">Mode</div>
      <div class="row">
        <button id="bTrack" class="active" title="Tracking ON: camera actively follows the locked target. Tracking OFF: freeze pan/tilt but keep the lock — does NOT clear or switch targets.">Tracking: ON</button>
        <button id="mHome" title="return to home pose">Home</button>
        <button id="mStop" style="background:#5a2126;border-color:#a23a3a" title="freeze in place">STOP</button>
        <button id="clear" title="forget current target">Clear</button>
        <button id="bPaint" style="background:#1f4d8c;border-color:#3a78c0" title="trigger one paint pulse (key P)">PAINT</button>
        <button id="bPaintAuto" title="auto-fire paint when selected target stays centered (stays on the SAME target)">Auto Paint: OFF</button>
        <button id="bSweep" title="Autonomous loop: pick a target → center on it → hold 2 s → PAINT → mark its id painted → advance to NEXT instance (per det_id, not per class). Forces Tracking ON.">Auto Serial Tracker: OFF</button>
        <button id="bSweepReset" title="forget painted memory">Reset memory</button>
        <button id="bMoveTgt" title="bounce the yellow ball in place">Move ball: OFF</button>
        <button id="bMoveCars" title="drive the prius+pickup in circles in front of the mantis">Move cars: OFF</button>
        <button id="bMoveBus" title="oscillate the inline red bus front/back along its body-x axis (north of mantis)">Move bus: OFF</button>
        <button id="bMoveMantis" title="ego-motion test: teleport the whole mantis chassis along world +x; tracker must hold lock through camera motion">Move turret: OFF</button>
        <label class="label" style="display:flex;align-items:center;gap:6px;margin-left:4px" title="V-key drive top speed (m/s). Affects both keyboard drive and the Move turret auto-sway amplitude.">
          Drive
          <input id="driveSpeed" type="range" min="1" max="25" step="0.5" value="4" style="width:120px;vertical-align:middle">
          <span id="driveSpeedV" style="min-width:36px;text-align:right">4.0</span> m/s
        </label>
        <button id="bWalkPpl" title="walk the two persons forward + slight turn">Walk people: OFF</button>
        <button id="bAutoZoom" title="auto-zoom in when locked target gets small, zoom out when it fills the frame">Auto Zoom: OFF</button>
        <button id="bKeyboard" title="enable keyboard control (K toggles). Arrows = jog. Space = PAINT. T = Tracking. C = Clear. H = Home. A = Auto. S = Stop. L = Lock center. Shift+Up/Down = Zoom. Esc = STOP.">Keyboard: OFF</button>
      </div>

      <div class="sect-head">Detector + Tracker</div>
      <div class="row">
        <button id="dAuto" class="active" title="YOLOv12 detection + ByteTrack ID-stable tracker (Kalman + IoU)">YOLOv12 + ByteTrack</button>
        <button id="dWorld" title="YOLO-World open-vocabulary detector — types any class as text (red truck, drone, ...). Set classes in the input below.">YOLO-World</button>
        <button id="dColor" title="HSV color detection + name+nearest-anchor tracker">HSV Color + AnchorMatch</button>
        <button id="bClickAim" title="clicks aim camera at pixel instead of selecting a target">Click-to-Aim: OFF</button>
        <span class="label" id="yoloStatus" style="align-self:center">init</span>
      </div>
      <div class="row" style="padding-top:0;align-items:center">
        <label class="label" style="flex:1">Open-vocab classes (comma-sep)
          <input id="worldClasses" type="text" value="car, person, truck, drone"
                 style="width:100%;height:28px;background:#1d2227;color:var(--text);border:1px solid #39434e;border-radius:6px;padding:0 8px;margin-top:4px">
        </label>
        <button id="bWorldApply" style="margin-top:18px">Apply</button>
      </div>

      <div class="sect-head">Jog (Manual or Auto override)</div>
      <div class="pad">
        <span class="sp"></span><button data-jog="tilt-up">&uarr;</button><span class="sp"></span>
        <button data-jog="pan-left">&larr;</button><button data-jog="center">&middot;</button><button data-jog="pan-right">&rarr;</button>
        <span class="sp"></span><button data-jog="tilt-down">&darr;</button><span class="sp"></span>
      </div>
      <div class="row" style="padding-top:0">
        <label class="label">Step&deg;
          <select id="step">
            <option value="0.5">0.5</option>
            <option value="1" selected>1</option>
            <option value="2">2</option>
            <option value="5">5</option>
            <option value="10">10</option>
          </select>
        </label>
        <span class="label">When Keyboard ON: Arrows/WASD=jog · Space=PAINT · T=Tracking · C=Clear · H=Home · Esc=STOP</span>
      </div>

      <div class="sect-head">Tracking tuning (slide to apply live)</div>
      <div class="gain" title="how aggressively the camera chases the target. Above ~1.0 the tilt axis overshoots and you'll see vertical bobble."><span>Speed</span><input id="gKp" type="range" min="0.05" max="1.0" step="0.01" value="0.55"><span id="gKpV">0.55</span></div>
      <div class="gain" title="corrects steady-state offset so target ends up exactly centered"><span>Hold</span><input id="gKi" type="range" min="0.00" max="0.60" step="0.01" value="0.20"><span id="gKiV">0.20</span></div>
      <div class="gain" title="damping; higher = smoother but slower to settle"><span>Smooth</span><input id="gKd" type="range" min="0.00" max="0.30" step="0.005" value="0.04"><span id="gKdV">0.04</span></div>
      <div class="gain" title="max degrees per second the camera can slew"><span>Max slew</span><input id="gRate" type="range" min="5" max="80" step="1" value="25"><span id="gRateV">25</span></div>
      <div class="gain" title="once target is this close to center, controller locks (no micro-jitter)"><span>Lock zone</span><input id="gDead" type="range" min="0.005" max="0.10" step="0.001" value="0.012"><span id="gDeadV">0.012</span></div>
      <div class="row">
        <button id="resetGains">Reset</button>
        <button id="autoTune">Auto-tune</button>
        <span class="label" id="autoStatus" style="align-self:center"></span>
      </div>
    </section>

    <section style="margin-top:12px">
      <div class="title"><span>Detections</span><span>click image to select</span></div>
      <table><thead><tr><th>ID</th><th>Name</th><th>Score</th></tr></thead><tbody id="detRows"></tbody></table>
    </section>
    <section style="margin-top:12px">
      <div class="title"><span>Virtual Marks</span><span>center-hold events</span></div>
      <table><thead><tr><th>Target</th><th>Pan</th><th>Tilt</th></tr></thead><tbody id="marks"></tbody></table>
    </section>
  </div>
</main>
<script>
const feed=document.getElementById('feed');
const stepSel=document.getElementById('step');
function setMode(m){
  if(m==='stop'){
    fetch('/api/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
  }else{
    fetch('/api/mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:m})});
  }
  modeBadge.textContent=m;
}
let trackingOn=true;
function setTrackingUI(){
  bTrack.textContent='Tracking: '+(trackingOn?'ON':'OFF');
  bTrack.classList.toggle('active',trackingOn);
}
bTrack.onclick=()=>{
  trackingOn=!trackingOn;
  setTrackingUI();
  setMode(trackingOn?'auto':'manual');
};
mHome.onclick=()=>setMode('home');
mStop.onclick=()=>setMode('stop');
bPaint.onclick=async ()=>{
  bPaint.disabled=true;
  try{ await fetch('/api/paint',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pulse_ms:120})}); }
  catch(e){}
  setTimeout(()=>bPaint.disabled=false, 250);
};
let autoPaintOn=false;
bPaintAuto.onclick=async ()=>{
  autoPaintOn=!autoPaintOn;
  bPaintAuto.textContent='Auto Paint: '+(autoPaintOn?'ON':'OFF');
  bPaintAuto.classList.toggle('active',autoPaintOn);
  await fetch('/api/paint_auto',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:autoPaintOn})});
};
let sweepOn=false;
bSweep.onclick=async ()=>{
  sweepOn=!sweepOn;
  bSweep.textContent='Auto Serial Tracker: '+(sweepOn?'ON':'OFF');
  bSweep.classList.toggle('active',sweepOn);
  // Sweep requires Tracking ON (mode=auto); reflect that in the UI so the
  // two buttons stop disagreeing.
  if(sweepOn && !trackingOn){
    trackingOn=true;
    bTrack.textContent='Tracking: ON';
    bTrack.classList.toggle('active',true);
  }
  await fetch('/api/sweep',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:sweepOn})});
};
bSweepReset.onclick=async ()=>{
  await fetch('/api/sweep',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reset_painted:true})});
};
let moveTgtOn=false;
bMoveTgt.onclick=async ()=>{
  moveTgtOn=!moveTgtOn;
  bMoveTgt.textContent='Move ball: '+(moveTgtOn?'ON':'OFF');
  bMoveTgt.classList.toggle('active',moveTgtOn);
  await fetch('/api/moving_target',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:moveTgtOn})});
};
let moveCarsOn=false;
bMoveCars.onclick=async ()=>{
  moveCarsOn=!moveCarsOn;
  bMoveCars.textContent='Move cars: '+(moveCarsOn?'ON':'OFF');
  bMoveCars.classList.toggle('active',moveCarsOn);
  await fetch('/api/cars_moving',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:moveCarsOn})});
};
let walkPplOn=false;
bWalkPpl.onclick=async ()=>{
  walkPplOn=!walkPplOn;
  bWalkPpl.textContent='Walk people: '+(walkPplOn?'ON':'OFF');
  bWalkPpl.classList.toggle('active',walkPplOn);
  await fetch('/api/people_walking',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:walkPplOn})});
};
let busMovingOn=false;
bMoveBus.onclick=async ()=>{
  busMovingOn=!busMovingOn;
  bMoveBus.textContent='Move bus: '+(busMovingOn?'ON':'OFF');
  bMoveBus.classList.toggle('active',busMovingOn);
  await fetch('/api/bus_moving',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:busMovingOn})});
};
let mantisMovingOn=false;
bMoveMantis.onclick=async ()=>{
  mantisMovingOn=!mantisMovingOn;
  bMoveMantis.textContent='Move turret: '+(mantisMovingOn?'ON':'OFF');
  bMoveMantis.classList.toggle('active',mantisMovingOn);
  await fetch('/api/mantis_moving',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:mantisMovingOn})});
};
driveSpeed.addEventListener('input',()=>{
  driveSpeedV.textContent=parseFloat(driveSpeed.value).toFixed(1);
  fetch('/api/mantis_speed',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({speed:parseFloat(driveSpeed.value)})});
});
let autoZoomOn=false;
bAutoZoom.onclick=async ()=>{
  autoZoomOn=!autoZoomOn;
  bAutoZoom.textContent='Auto Zoom: '+(autoZoomOn?'ON':'OFF');
  bAutoZoom.classList.toggle('active',autoZoomOn);
  await fetch('/api/auto_zoom',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:autoZoomOn})});
};
document.addEventListener('keypress',e=>{
  if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT') return;
  if(e.key==='p'||e.key==='P'){ bPaint.click(); }
});

// Connect dropdown: switch frame source between Gazebo and webcams.
const connectMenu=document.getElementById('connectMenu');
bConnect.onclick=(e)=>{
  e.stopPropagation();
  connectMenu.style.display = (connectMenu.style.display==='none' ? 'block' : 'none');
};
document.addEventListener('click',(e)=>{
  if(!connectMenu.contains(e.target) && e.target!==bConnect){
    connectMenu.style.display='none';
  }
});
document.querySelectorAll('.connOpt').forEach(b=>{
  b.onclick=async ()=>{
    const src=b.dataset.src;
    bConnect.textContent='Connect: probing…';
    const r=await fetch('/api/source',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source:src})}).then(r=>r.json());
    if(r.ok){
      const resolved = r.source;
      const label = resolved==='gz' ? 'Gazebo Sim' :
                    resolved.startsWith('webcam:') ? ('Webcam #'+resolved.split(':')[1]) : resolved;
      bConnect.textContent='Connect: '+label;
      if(r.detected){ console.log('auto-detect:', r.detected); }
    } else {
      bConnect.textContent='Connect: failed';
      alert('Connect failed: '+(r.error||r.message||'unknown'));
    }
    connectMenu.style.display='none';
  };
});

function setDetector(d){
  fetch('/api/detector',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:d})});
  dAuto.classList.toggle('active',d==='auto');
  dWorld.classList.toggle('active',d==='world');
  dColor.classList.toggle('active',d==='color');
}
dAuto.onclick=()=>setDetector('auto');
dWorld.onclick=()=>setDetector('world');
dColor.onclick=()=>setDetector('color');
bWorldApply.onclick=async ()=>{
  const txt=document.getElementById('worldClasses').value;
  await fetch('/api/world_classes',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({classes:txt})});
};

let clickAim=false;
let zoomVal=1.0;
const zoomEl=document.getElementById('zoom'),zoomLbl=document.getElementById('zoomV');
let zoomPostTimer=null;
function applyZoom(){
  zoomVal=parseFloat(zoomEl.value);
  zoomLbl.textContent=zoomVal.toFixed(1)+'x';
  clearTimeout(zoomPostTimer);
  zoomPostTimer=setTimeout(()=>{
    fetch('/api/zoom',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({zoom:zoomVal})});
  },80);
}
zoomEl.addEventListener('input',applyZoom);
applyZoom();
bClickAim.onclick=()=>{
  clickAim=!clickAim;
  bClickAim.textContent='Click-to-Aim: '+(clickAim?'ON':'OFF');
  bClickAim.classList.toggle('active',clickAim);
};
feed.addEventListener('click', async e=>{
  const r=feed.getBoundingClientRect();
  // Server zooms before delivering JPG, so the displayed pixel coords ARE
  // the zoomed-frame coords. Direct map to 1280x720.
  const x=(e.clientX-r.left)/r.width*1280;
  const y=(e.clientY-r.top)/r.height*720;
  const url=clickAim?'/api/click_target':'/api/select';
  await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({x,y})});
});
clear.onclick=()=>fetch('/api/select',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({clear:true})});

function jog(dir){
  const step=parseFloat(stepSel.value);
  let dp=0,dt=0;
  if(dir==='pan-left') dp=+step;
  if(dir==='pan-right') dp=-step;
  if(dir==='tilt-up') dt=-step;
  if(dir==='tilt-down') dt=+step;
  if(dir==='center'){ fetch('/api/jog',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({home:true})}); return; }
  fetch('/api/jog',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dpan:dp,dtilt:dt})});
}
document.querySelectorAll('[data-jog]').forEach(b=>b.onclick=()=>jog(b.dataset.jog));

// Restore client-only flags from localStorage so a browser refresh keeps
// the user's previous keyboard+drive choices.
let keyboardOn=localStorage.getItem('mantis_keyboardOn')==='1';
bKeyboard.textContent='Keyboard: '+(keyboardOn?'ON':'OFF');
bKeyboard.classList.toggle('active',keyboardOn);
bKeyboard.onclick=()=>{
  keyboardOn=!keyboardOn;
  bKeyboard.textContent='Keyboard: '+(keyboardOn?'ON':'OFF');
  bKeyboard.classList.toggle('active',keyboardOn);
  localStorage.setItem('mantis_keyboardOn', keyboardOn?'1':'0');
};
window._driveMode = localStorage.getItem('mantis_driveMode') || 'mantis';
if(window._driveMode==='vehicle') bConnect.style.borderColor='var(--amber)';
function nudgeZoom(delta){
  const mn=parseFloat(zoomEl.min),mx=parseFloat(zoomEl.max);
  let v=parseFloat(zoomEl.value)+delta;
  v=Math.max(mn,Math.min(mx,Math.round(v*10)/10));
  zoomEl.value=v;
  applyZoom();
}
async function lockBestInView(){
  // Pick the detection nearest the image-center crosshair so 'L' locks
  // whatever the camera is aimed at, not whichever target YOLO happens to
  // score highest.
  const r=await fetch('/api/status'); const s=await r.json();
  const dets=(s.detections||[]).filter(d=>d.score>=0.10);
  if(!dets.length) return;
  const W=1280,H=720,cxImg=W/2,cyImg=H/2;
  const scored=dets.map(d=>{
    const cx=(d.bbox[0]+d.bbox[2])/2, cy=(d.bbox[1]+d.bbox[3])/2;
    return {d,cx,cy,dist:Math.hypot(cx-cxImg,cy-cyImg)};
  });
  scored.sort((a,b)=>a.dist-b.dist);
  const t=scored[0];
  fetch('/api/select',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({x:t.cx,y:t.cy})});
}
document.addEventListener('keyup',e=>{
  if(window._driveMode!=='vehicle') return;
  const k=e.key.toLowerCase();
  if(['arrowup','arrowdown','w'].includes(k)){
    fetch('/api/mantis_drive',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({vx:0})});
  } else if(['arrowleft','arrowright','d'].includes(k)){
    fetch('/api/mantis_drive',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({vyaw:0})});
  }
});
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT'||e.target.tagName==='TEXTAREA') return;
  // Zoom + mode/lock shortcuts work without keyboard mode being toggled on.
  if((e.shiftKey||e.ctrlKey)&&(e.key==='ArrowUp'||e.key==='ArrowDown')){
    nudgeZoom(e.key==='ArrowUp'?+0.2:-0.2);
    e.preventDefault();
    return;
  }
  const k=e.key.toLowerCase();
  // Global mode/lock keys — active regardless of keyboard-jog toggle.
  if(k==='c'){ clear.click(); return; }
  if(k==='a'){ setMode('auto'); return; }
  if(k==='s'){ setMode('stop'); return; }
  if(k==='h'){ setMode('home'); return; }
  if(k==='l'){ lockBestInView(); return; }
  if(k==='k'){ bKeyboard.click(); return; }
  if(k==='v'){ window._driveMode='vehicle'; localStorage.setItem('mantis_driveMode','vehicle'); bConnect.style.borderColor='var(--amber)'; console.log('keyboard drive: VEHICLE (front/back/left/right)'); return; }
  if(k==='m'){ window._driveMode='mantis'; localStorage.setItem('mantis_driveMode','mantis'); bConnect.style.borderColor=''; console.log('keyboard drive: MANTIS (pan/tilt)'); return; }
  if(k==='escape'||k==='x'){ setMode('stop'); e.preventDefault(); return; }
  if(!keyboardOn) return;
  if(window._driveMode==='vehicle'){
    const SPEED=parseFloat(driveSpeed.value)||4.0;
    const YAW_SPEED=Math.min(2.5, SPEED * 0.25);
    // Press = start drive (velocity setpoint). Release handled by keyup.
    if(e.repeat){ e.preventDefault(); return; }
    let vx=null,vy=null,vyaw=null;
    if(k==='arrowup'||k==='w') vx=+SPEED;
    else if(k==='arrowdown') vx=-SPEED;
    else if(k==='arrowleft') vyaw=+YAW_SPEED;
    else if(k==='arrowright'||k==='d') vyaw=-YAW_SPEED;
    if(vx!==null||vy!==null||vyaw!==null){
      const body={};
      if(vx!==null) body.vx=vx;
      if(vy!==null) body.vy=vy;
      if(vyaw!==null) body.vyaw=vyaw;
      fetch('/api/mantis_drive',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
      e.preventDefault();
      return;
    }
  }
  if(k==='arrowleft'){jog('pan-left');e.preventDefault();}
  else if(k==='arrowright'||k==='d'){jog('pan-right');e.preventDefault();}
  else if(k==='arrowup'||k==='w'){jog('tilt-up');e.preventDefault();}
  else if(k==='arrowdown'){jog('tilt-down');e.preventDefault();}
  else if(k===' '){bPaint.click();e.preventDefault();}
  else if(k==='t'){bTrack.click();}
  else if(k==='p'){bPaint.click();}
});

function bindGain(id,key,vid){
  const el=document.getElementById(id),v=document.getElementById(vid);
  const update=()=>{
    window._slidersDirty=true;
    clearTimeout(window._slidersClean);
    window._slidersClean=setTimeout(()=>{window._slidersDirty=false;},1500);
    const f=parseFloat(el.value);
    const dec=parseFloat(el.step)<0.01?3:(parseFloat(el.step)<1?2:0);
    v.textContent=f.toFixed(dec);
    const body={}; body[key]=f;
    fetch('/api/gains',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  };
  el.addEventListener('input',update);
  el.addEventListener('change',update);
}
bindGain('gKp','kp','gKpV');
bindGain('gKi','ki','gKiV');
bindGain('gKd','kd','gKdV');
bindGain('gRate','max_rate','gRateV');
bindGain('gDead','deadband','gDeadV');
document.getElementById('resetGains').onclick=()=>{
  fetch('/api/gains',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reset:true})}).then(()=>location.reload());
};
document.getElementById('autoTune').onclick=async ()=>{
  autoStatus.textContent='running ~12s ...';
  autoTune.disabled=true;
  try{
    const r=await fetch('/api/autotune',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}).then(r=>r.json());
    if(r.ok){
      autoStatus.textContent=`done: Speed=${r.kp.toFixed(2)} Hold=${r.ki.toFixed(2)} Smooth=${r.kd.toFixed(2)}`;
      setTimeout(()=>location.reload(),1500);
    }else{
      autoStatus.textContent='failed: '+(r.reason||'unknown');
    }
  }catch(e){ autoStatus.textContent='error: '+e; }
  autoTune.disabled=false;
};

async function poll(){
  try{
    const s=await fetch('/api/status').then(r=>r.json());
    status.textContent=s.camera_age_s<1.5?'live':'waiting for Gazebo camera';
    selected.textContent=s.selected_name||s.selected_id||'none';
    detCount.textContent=s.detections.length;
    pan.textContent=s.pan_deg.toFixed(1);
    tilt.textContent=s.tilt_deg.toFixed(1);
    modeBadge.textContent=s.mode+(s.sweep_enabled?' (sweep)':'');
    trackingOn = (s.mode==='auto');
    setTrackingUI();
    yoloStatus.textContent=s.yolo_status||'';
    dAuto.classList.toggle('active',s.detector_mode==='auto');
    dWorld.classList.toggle('active',s.detector_mode==='world');
    dColor.classList.toggle('active',s.detector_mode==='color');
    if(s.gains && !window._slidersDirty){
      gKp.value=s.gains.kp; gKpV.textContent=s.gains.kp.toFixed(2);
      gKi.value=s.gains.ki; gKiV.textContent=s.gains.ki.toFixed(2);
      gKd.value=s.gains.kd; gKdV.textContent=s.gains.kd.toFixed(3);
      gRate.value=s.gains.max_rate; gRateV.textContent=s.gains.max_rate.toFixed(0);
      gDead.value=s.gains.deadband; gDeadV.textContent=s.gains.deadband.toFixed(3);
    }
    if(s.sweep_enabled!==undefined){
      sweepOn=!!s.sweep_enabled;
      const tail=s.sweep_painted&&s.sweep_painted.length?` [${s.sweep_painted.length} done]`:'';
      bSweep.textContent='Auto Serial Tracker: '+(sweepOn?'ON':'OFF')+tail;
      bSweep.classList.toggle('active',sweepOn);
    }
    if(s.auto_zoom!==undefined){
      autoZoomOn=!!s.auto_zoom;
      bAutoZoom.textContent='Auto Zoom: '+(autoZoomOn?'ON':'OFF');
      bAutoZoom.classList.toggle('active',autoZoomOn);
    }
    if(s.frame_source){
      const lbl = s.frame_source==='gz' ? 'Gazebo Sim' :
                  s.frame_source.startsWith('webcam:') ? ('Webcam #'+s.frame_source.split(':')[1]) : s.frame_source;
      bConnect.textContent='Connect: '+lbl;
    }
    // Sync the rest of the toggles + slider from server state so a browser
    // refresh doesn't reset what the user already enabled.
    if(s.cars_moving!==undefined){
      moveCarsOn=!!s.cars_moving;
      bMoveCars.textContent='Move cars: '+(moveCarsOn?'ON':'OFF');
      bMoveCars.classList.toggle('active',moveCarsOn);
    }
    if(s.people_walking!==undefined){
      walkPplOn=!!s.people_walking;
      bWalkPpl.textContent='Walk people: '+(walkPplOn?'ON':'OFF');
      bWalkPpl.classList.toggle('active',walkPplOn);
    }
    if(s.moving_target!==undefined){
      moveTgtOn=!!s.moving_target;
      bMoveTgt.textContent='Move ball: '+(moveTgtOn?'ON':'OFF');
      bMoveTgt.classList.toggle('active',moveTgtOn);
    }
    if(s.bus_moving!==undefined){
      busMovingOn=!!s.bus_moving;
      bMoveBus.textContent='Move bus: '+(busMovingOn?'ON':'OFF');
      bMoveBus.classList.toggle('active',busMovingOn);
    }
    if(s.mantis_moving!==undefined){
      mantisMovingOn=!!s.mantis_moving;
      bMoveMantis.textContent='Move turret: '+(mantisMovingOn?'ON':'OFF');
      bMoveMantis.classList.toggle('active',mantisMovingOn);
    }
    if(s.mantis_drive_speed!==undefined && document.activeElement!==driveSpeed){
      driveSpeed.value=s.mantis_drive_speed;
      driveSpeedV.textContent=Number(s.mantis_drive_speed).toFixed(1);
    }
    detRows.innerHTML=s.detections.map(d=>`<tr><td><button onclick="selectDetection(${d.id})">${d.id}</button></td><td>${d.name}</td><td>${(d.score*100).toFixed(0)}%</td></tr>`).join('');
    marks.innerHTML=s.virtual_marks.slice(0,8).map(m=>`<tr><td>${m.target}</td><td>${m.pan_deg}</td><td>${m.tilt_deg}</td></tr>`).join('');
  }catch(e){}
}
async function selectDetection(id){
  await fetch('/api/select_detection',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id})});
}
setInterval(poll,500); poll();

let agentOn=false;
async function refreshAgentModels(){
  const r=await fetch('/api/agent/models').then(r=>r.json());
  const sel=document.getElementById('agentModel');
  const cur=sel.value;
  sel.innerHTML=(r.models||[]).map(m=>`<option value="${m}">${m}</option>`).join('');
  if(r.selected) sel.value=r.selected; else if(cur) sel.value=cur;
  agentOn=!!r.enabled;
  bAgentToggle.textContent='Agent: '+(agentOn?'ON':'OFF');
  bAgentToggle.classList.toggle('active',agentOn);
}
refreshAgentModels();
setInterval(refreshAgentModels, 15000);
bAgentToggle.onclick=async()=>{
  agentOn=!agentOn;
  const m=document.getElementById('agentModel').value || null;
  await fetch('/api/agent/enable',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:agentOn,model:m})});
  bAgentToggle.textContent='Agent: '+(agentOn?'ON':'OFF');
  bAgentToggle.classList.toggle('active',agentOn);
};
function appendChat(role,text){
  const log=document.getElementById('chatLog');
  const div=document.createElement('div');
  div.style.margin='4px 0';
  div.innerHTML=`<b style="color:${role==='user'?'#56cfe1':'#ffd166'}">${role}:</b> ${text}`;
  log.appendChild(div); log.scrollTop=log.scrollHeight;
}
chatSend.onclick=async()=>{
  const t=chatIn.value.trim(); if(!t) return;
  chatIn.value=''; appendChat('user',t);
  if(!agentOn){ appendChat('system','agent off; toggle on'); return; }
  const r=await fetch('/api/agent/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:t})}).then(r=>r.json());
  if(r.ok){
    appendChat('agent', (r.reply||'').replace(/\n/g,'<br>'));
    if(r.action) appendChat('action', `${r.action} -> ${r.action_result}`);
  } else { appendChat('error', r.error||'unknown'); }
};
chatIn.addEventListener('keydown',e=>{ if(e.key==='Enter'){ chatSend.click(); }});

async function pushChannels(){
  const body={
    channels:{gz_topic:chGz.checked,file:chFile.checked,udp:chUdp.checked,tcp:chTcp.checked,serial:chSerial.checked},
    udp:{host:udpHost.value,port:parseInt(udpPort.value)||9000},
    tcp:{host:tcpHost.value,port:parseInt(tcpPort.value)||9001},
    serial:{port:serPort.value,baud:parseInt(serBaud.value)||9600},
  };
  await fetch('/api/channels',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
}
chSave.onclick=pushChannels;
[chGz,chFile,chUdp,chTcp,chSerial].forEach(el=>el.addEventListener('change',pushChannels));
fetch('/api/channels').then(r=>r.json()).then(c=>{
  chGz.checked=c.channels.gz_topic; chFile.checked=c.channels.file;
  chUdp.checked=c.channels.udp; chTcp.checked=c.channels.tcp; chSerial.checked=c.channels.serial;
  udpHost.value=c.udp.host; udpPort.value=c.udp.port;
  tcpHost.value=c.tcp.host; tcpPort.value=c.tcp.port;
  serPort.value=c.serial.port; serBaud.value=c.serial.baud;
});
</script>
</body>
</html>
"""


@app.get("/")
def index() -> Response:
    return Response(HTML_PAGE, mimetype="text/html")


@app.get("/video_feed")
def video_feed():
    def gen():
        last_sent_stamp = 0.0
        while True:
            jpg = latest_annotated
            stamp = latest_stamp
            if jpg is None:
                blank = np.zeros((720, 1280, 3), np.uint8)
                cv2.putText(blank, "Waiting for /mantis/nose_camera/image",
                            (340, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (240, 240, 240), 2)
                _, encoded = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 60])
                jpg = encoded.tobytes()
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(0.10)
                continue
            if stamp != last_sent_stamp:
                last_sent_stamp = stamp
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            else:
                time.sleep(0.015)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/health")
def api_health():
    now = time.time()
    cam_age = (now - latest_stamp) if latest_stamp else float("inf")
    joint_age = (now - joint_state_stamp) if joint_state_stamp else float("inf")
    issues = []
    if cam_age > CAMERA_STALE_S:
        issues.append(f"camera stale {cam_age:.1f}s")
    if joint_age > JOINT_STALE_S:
        issues.append(f"joint_state stale {joint_age:.1f}s")
    if yolo_status.startswith(("load", "ultralytics", "detector exception", "track failed")):
        if "loaded" not in yolo_status:
            issues.append(f"detector: {yolo_status}")
    ok = len(issues) == 0
    return jsonify({
        "ok": ok,
        "camera_age_s": cam_age if cam_age != float("inf") else None,
        "joint_age_s": joint_age if joint_age != float("inf") else None,
        "frame_count": frame_count,
        "paint_count": paint_count,
        "mode": mode,
        "issues": issues,
    }), (200 if ok else 503)


@app.get("/api/status")
def api_status():
    with lock:
        visible_detections = detections
        if not visible_detections and time.time() - recent_detection_stamp < 1.0:
            visible_detections = recent_detections
        dets = [
            {"id": d.det_id, "name": d.name, "bbox": d.bbox, "score": d.score}
            for d in visible_detections
        ]
        return jsonify({
            "camera_age_s": time.time() - latest_stamp if latest_stamp else 999,
            "frame_count": frame_count,
            "detections": dets,
            "selected_id": selected_id,
            "selected_name": selected_name,
            "pan_deg": pan_deg,
            "tilt_deg": tilt_deg,
            "actual_pan_deg": actual_pan_deg,
            "actual_tilt_deg": actual_tilt_deg,
            "pan_vel_deg_s": pan_vel_deg_s,
            "tilt_vel_deg_s": tilt_vel_deg_s,
            "target_vx_pix_s": target_vx_pix_s,
            "target_vy_pix_s": target_vy_pix_s,
            "mode": mode,
            "centered_frames": centered_frames,
            "virtual_marks": virtual_marks,
            "gains": {
                "kp": pan_gains.kp, "ki": pan_gains.ki, "kd": pan_gains.kd,
                "max_rate": pan_gains.max_rate_deg_s,
                "deadband": pan_gains.deadband_norm,
            },
            "detector_mode": detector_mode,
            "tracker": ("ByteTrack" if detector_mode == "auto" else "AnchorMatch"),
            "yolo_status": yolo_status,
            "detection_latency_ms": int(_detection_last_latency_s * 1000),
            "detection_age_s": (time.time() - _detection_last_completed)
                               if _detection_last_completed else None,
            "zoom": zoom_factor,
            "auto_zoom": auto_zoom_enabled,
            "frame_source": frame_source,
            "eff_hfov_deg": math.degrees(_eff_hfov),
            "eff_vfov_deg": math.degrees(_eff_vfov),
            "paint_count": paint_count,
            "paint_auto": paint_auto,
            "sweep_enabled": sweep_enabled,
            "sweep_painted": sorted(sweep_painted_ids),
            "cars_moving": cars_moving,
            "people_walking": people_walking,
            "moving_target": target_moving,
            "bus_moving": bus_moving,
            "mantis_moving": mantis_moving,
            "mantis_drive_speed": mantis_drive_speed,
        })


CLICK_MAX_NEAREST_PX = 120.0


@app.post("/api/select")
def api_select():
    global selected_id, selected_name, selected_anchor_xy, mode
    data = request.get_json(force=True, silent=True) or {}
    with lock:
        if data.get("clear"):
            clear_selection()
            mode = "home"
            return jsonify({"ok": True, "selected_id": selected_id,
                            "selected_name": selected_name, "mode": mode})

        x = safe_float(data, "x", -1.0, -1.0, IMG_W * 2.0)
        y = safe_float(data, "y", -1.0, -1.0, IMG_H * 2.0)
        selectable = detections
        if not selectable and time.time() - recent_detection_stamp < 1.0:
            selectable = recent_detections
        if not selectable:
            return jsonify({"ok": False, "reason": "no detections",
                            "selected_id": selected_id,
                            "selected_name": selected_name,
                            "mode": mode})

        # 1) hit-test: click inside any bbox (small padding)
        pad = 8
        hit = None
        for d in selectable:
            x1, y1, x2, y2 = d.bbox
            if x1 - pad <= x <= x2 + pad and y1 - pad <= y <= y2 + pad:
                hit = d
                break
        # 2) fallback: nearest bbox center within tolerance
        if hit is None:
            best = min(selectable, key=lambda d: (
                ((d.bbox[0] + d.bbox[2]) / 2 - x) ** 2
                + ((d.bbox[1] + d.bbox[3]) / 2 - y) ** 2
            ))
            bcx = (best.bbox[0] + best.bbox[2]) / 2
            bcy = (best.bbox[1] + best.bbox[3]) / 2
            if math.hypot(bcx - x, bcy - y) <= CLICK_MAX_NEAREST_PX:
                hit = best

        if hit is None:
            # No match — KEEP current selection (don't wipe it on a miss).
            return jsonify({"ok": False, "reason": "no target near click",
                            "selected_id": selected_id,
                            "selected_name": selected_name,
                            "mode": mode})

        # Only NOW reset transient controller state for the new selection.
        clear_selection()
        # Sync cmd to actual joint position so lead window doesn't yank the
        # camera back to where the previous cmd left off.
        global pan_deg, tilt_deg, jog_pan_target, jog_tilt_target
        if joint_state_stamp:
            pan_deg = actual_pan_deg
            tilt_deg = actual_tilt_deg
        jog_pan_target = None
        jog_tilt_target = None
        x1, y1, x2, y2 = hit.bbox
        global selected_signature, selected_signature_anchor, selected_kf
        selected_id = hit.det_id
        selected_name = hit.name
        ax = (x1 + x2) / 2; ay = (y1 + y2) / 2
        selected_anchor_xy = (ax, ay)
        sig0 = _hsv_signature_from_bbox(latest_raw, hit.bbox)
        selected_signature = sig0
        selected_signature_anchor = sig0  # immutable lock-time copy
        selected_kf = TargetKF(ax, ay)
        mode = "auto"
    return jsonify({"ok": True, "selected_id": selected_id,
                    "selected_name": selected_name, "mode": mode})


@app.post("/api/select_detection")
def api_select_detection():
    global selected_id, selected_name, selected_anchor_xy, mode
    data = request.get_json(force=True, silent=True) or {}
    det_id = safe_int(data, "id", 0, -1, 10**6)
    with lock:
        selectable = detections
        if not selectable and time.time() - recent_detection_stamp < 1.0:
            selectable = recent_detections
        hit = next((d for d in selectable if d.det_id == det_id), None)
        if hit is None:
            return jsonify({"ok": False, "reason": "id not in detections",
                            "selected_id": selected_id,
                            "selected_name": selected_name, "mode": mode})
        clear_selection()
        global pan_deg, tilt_deg, jog_pan_target, jog_tilt_target
        if joint_state_stamp:
            pan_deg = actual_pan_deg
            tilt_deg = actual_tilt_deg
        jog_pan_target = None
        jog_tilt_target = None
        global selected_signature, selected_signature_anchor, selected_kf
        selected_id = hit.det_id
        selected_name = hit.name
        x1, y1, x2, y2 = hit.bbox
        ax = (x1 + x2) / 2; ay = (y1 + y2) / 2
        selected_anchor_xy = (ax, ay)
        sig0 = _hsv_signature_from_bbox(latest_raw, hit.bbox)
        selected_signature = sig0
        selected_signature_anchor = sig0
        selected_kf = TargetKF(ax, ay)
        mode = "auto"
    return jsonify({"ok": True, "selected_id": selected_id,
                    "selected_name": selected_name, "mode": mode})


@app.post("/api/mode")
def api_mode():
    global mode
    data = request.get_json(force=True, silent=True) or {}
    new_mode = str(data.get("mode", "")).lower()
    if new_mode not in ("auto", "manual", "home", "stop", "passthrough"):
        return jsonify({"ok": False, "message": "mode must be auto|manual|home|stop|passthrough"}), 400
    with lock:
        if new_mode != "auto":
            clear_selection()
        mode = new_mode
    return jsonify({"ok": True, "mode": mode})


@app.post("/api/jog")
def api_jog():
    global mode, jog_pan_target, jog_tilt_target, pan_deg, tilt_deg
    data = request.get_json(force=True, silent=True) or {}
    with lock:
        if data.get("home"):
            jog_pan_target = HOME_PAN_DEG
            jog_tilt_target = HOME_TILT_DEG
            mode = "manual"
            clear_selection()
            return jsonify({"ok": True, "mode": mode})
        dpan = safe_float(data, "dpan", 0.0, -180.0, 180.0)
        dtilt = safe_float(data, "dtilt", 0.0, -90.0, 90.0)
        base_pan = jog_pan_target if jog_pan_target is not None else pan_deg
        base_tilt = jog_tilt_target if jog_tilt_target is not None else tilt_deg
        jog_pan_target = clamp(base_pan + dpan, PAN_LIMIT[0], PAN_LIMIT[1])
        jog_tilt_target = clamp(base_tilt + dtilt, TILT_LIMIT[0], TILT_LIMIT[1])
        clear_selection()
        mode = "manual"
    return jsonify({"ok": True, "mode": mode,
                    "pan_target": jog_pan_target, "tilt_target": jog_tilt_target})


@app.post("/api/gains")
def api_gains():
    data = request.get_json(force=True, silent=True) or {}
    with lock:
        if data.get("reset"):
            # Higher max_rate so the nose can keep up with the chassis (V
            # drive) and bus motion; tilt gets a separate cap because the
            # joint is bandwidth-limited.
            pan_gains.kp = 0.65; pan_gains.ki = 0.20; pan_gains.kd = 0.15
            pan_gains.max_rate_deg_s = 60.0; pan_gains.deadband_norm = 0.008
            tilt_gains.kp = 0.60; tilt_gains.ki = 0.18; tilt_gains.kd = 0.15
            tilt_gains.max_rate_deg_s = 45.0; tilt_gains.deadband_norm = 0.012
            reset_controller_state()
            return jsonify({"ok": True, "reset": True})
        for key in ("kp", "ki", "kd"):
            if key in data:
                lo, hi = GAIN_BOUNDS[key]
                v = safe_float(data, key, getattr(pan_gains, key), lo, hi)
                setattr(pan_gains, key, v)
                setattr(tilt_gains, key, v)
        if "max_rate" in data:
            lo, hi = GAIN_BOUNDS["max_rate"]
            v = safe_float(data, "max_rate", pan_gains.max_rate_deg_s, lo, hi)
            pan_gains.max_rate_deg_s = v
            tilt_gains.max_rate_deg_s = clamp(v * 0.8, 1.0, hi)
        if "deadband" in data:
            lo, hi = GAIN_BOUNDS["deadband"]
            v = safe_float(data, "deadband", pan_gains.deadband_norm, lo, hi)
            pan_gains.deadband_norm = v
            tilt_gains.deadband_norm = clamp(v * 1.3, lo, hi)
    return jsonify({"ok": True})


def _autotune_axis(topic: str, axis: str, baseline_deg: float, step_deg: float,
                   settle_s: float = 2.0, capture_s: float = 3.0):
    pub = pan_pub if axis == "pan" else tilt_pub
    # hold baseline
    t_end = time.time() + settle_s
    while time.time() < t_end:
        publish_angle(pub, baseline_deg)
        time.sleep(0.02)
    # step
    samples: list[tuple[float, float]] = []
    t0 = time.time()
    while time.time() - t0 < capture_s:
        publish_angle(pub, baseline_deg + step_deg)
        pos_deg = actual_pan_deg if axis == "pan" else actual_tilt_deg
        samples.append((time.time() - t0, pos_deg))
        time.sleep(0.02)
    return samples


def _fopdt_fit(samples, baseline, target):
    if len(samples) < 10:
        return None
    span = target - baseline
    if abs(span) < 1e-3:
        return None
    final = sum(p for _, p in samples[-5:]) / 5.0
    K = (final - baseline) / span
    threshold = baseline + span * 0.632
    low = baseline + span * 0.05
    t_start = None
    t63 = None
    for t, p in samples:
        if t_start is None and ((span > 0 and p >= low) or (span < 0 and p <= low)):
            t_start = t
        if t63 is None and ((span > 0 and p >= threshold) or (span < 0 and p <= threshold)):
            t63 = t
            break
    if t_start is None or t63 is None or t63 <= t_start:
        return None
    return K, max(0.02, t63 - t_start), max(0.01, t_start)


def _cohen_coon(K, tau, theta):
    K = K if abs(K) > 1e-3 else 1.0
    r = theta / tau
    Kp = (1.20 / K) * (1.0 + 0.18 * r) / max(0.05, 1.0 - 0.39 * r)
    Ti = theta * (2.5 - 2.0 * r) / max(0.1, 1.0 - 0.39 * r)
    Td = theta * 0.37 / max(0.05, 1.0 - 0.81 * r)
    Ki = Kp / max(0.05, Ti)
    Kd = Kp * Td
    return Kp, Ki, Kd


@app.post("/api/autotune")
def api_autotune():
    global mode
    if not joint_state_stamp:
        return jsonify({"ok": False, "reason": "no joint_states topic"})
    with lock:
        prev_mode = mode
        clear_selection()
        mode = "passthrough"

    try:
        pan_samples = _autotune_axis(PAN_TOPIC, "pan", 0.0, 18.0)
        pan_fit = _fopdt_fit(pan_samples, 0.0, 18.0)
        tilt_samples = _autotune_axis(TILT_TOPIC, "tilt", 12.0, 10.0)
        tilt_fit = _fopdt_fit(tilt_samples, 12.0, 22.0)
    finally:
        with lock:
            mode = prev_mode if prev_mode != "passthrough" else "home"

    if pan_fit is None or tilt_fit is None:
        return jsonify({"ok": False, "reason": "FOPDT fit failed (joint did not respond)"})

    Kp_p, Ki_p, Kd_p = _cohen_coon(*pan_fit)
    Kp_t, Ki_t, Kd_t = _cohen_coon(*tilt_fit)
    Kp = max(0.15, min(1.2, (Kp_p + Kp_t) / 2.0))
    Ki = max(0.00, min(0.30, (Ki_p + Ki_t) / 2.0))
    Kd = max(0.00, min(0.20, (Kd_p + Kd_t) / 2.0))
    # Conservative: scale Cohen-Coon down so outer loop stays smooth in face
    # of YOLO latency + bbox jitter.
    Kp *= 0.45
    Ki *= 0.20
    Kd *= 0.40
    Kp = max(0.20, min(0.90, Kp))

    with lock:
        pan_gains.kp = Kp; pan_gains.ki = Ki; pan_gains.kd = Kd
        tilt_gains.kp = Kp; tilt_gains.ki = Ki; tilt_gains.kd = Kd

    return jsonify({"ok": True, "kp": Kp, "ki": Ki, "kd": Kd,
                    "pan_fit": {"K": pan_fit[0], "tau": pan_fit[1], "theta": pan_fit[2]},
                    "tilt_fit": {"K": tilt_fit[0], "tau": tilt_fit[1], "theta": tilt_fit[2]}})


def trigger_paint(reason: str, pulse_ms: int = PAINT_PULSE_MS_DEFAULT) -> dict:
    """Emit a virtual paint mark + hardware-out signal.

    Side-effects (all non-destructive, sim-only):
    - Increments paint_count.
    - Publishes Int32(pulse_ms) on /mantis/paint_trigger so any external
      subscriber (RPi GPIO/PWM bridge, MCU, ROS bridge) can react.
    - Writes a one-line record to PAINT_SIGNAL_FILE which a user-side daemon
      can tail to drive a real PWM pin.
    No physics, no projectile, no impact simulation in the world.
    """
    global paint_count, paint_last_ts
    now = time.time()
    paint_count += 1
    paint_last_ts = now
    record = {
        "n": paint_count,
        "time": round(now, 3),
        "reason": reason,
        "selected_id": selected_id,
        "selected_name": selected_name,
        "pan_deg": round(actual_pan_deg, 3),
        "tilt_deg": round(actual_tilt_deg, 3),
        "pulse_ms": int(pulse_ms),
    }
    virtual_marks.insert(0, record)
    del virtual_marks[64:]
    cx_mark, cy_mark = IMG_W / 2.0, IMG_H / 2.0
    if selected_id is not None or selected_name is not None:
        target = resolve_selected_target()
        if target is not None:
            x1, y1, x2, y2 = target.bbox
            cx_mark = (x1 + x2) / 2.0
            cy_mark = (y1 + y2) / 2.0
    paint_overlay_marks.append({
        "cx": cx_mark, "cy": cy_mark, "ts": now,
        "color": (40, 80, 240),
    })
    del paint_overlay_marks[32:]
    payload_line = (f"{record['time']} {paint_count} {pulse_ms} "
                    f"{actual_pan_deg:.3f} {actual_tilt_deg:.3f} "
                    f"{selected_name or '-'}\n")
    payload_bytes = payload_line.encode()
    if paint_channels.get("gz_topic"):
        try:
            msg = Int32()
            msg.data = int(pulse_ms)
            paint_pub.publish(msg)
        except Exception as exc:
            record["topic_err"] = str(exc)
    if paint_channels.get("file"):
        try:
            with open(PAINT_SIGNAL_FILE, "a") as f:
                f.write(payload_line)
        except Exception as exc:
            record["file_err"] = str(exc)
    if paint_channels.get("udp"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.5)
            s.sendto(payload_bytes, paint_udp_addr)
            s.close()
        except Exception as exc:
            record["udp_err"] = str(exc)
    if paint_channels.get("tcp"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect(paint_tcp_addr)
            s.sendall(payload_bytes)
            s.close()
        except Exception as exc:
            record["tcp_err"] = str(exc)
    if paint_channels.get("serial"):
        try:
            import serial as _ser  # type: ignore
            with _ser.Serial(paint_serial_port, paint_serial_baud,
                             timeout=0.3, write_timeout=0.3) as port:
                port.write(payload_bytes)
        except Exception as exc:
            record["serial_err"] = str(exc)
    return record


@app.post("/api/paint")
def api_paint():
    data = request.get_json(force=True, silent=True) or {}
    pulse = safe_int(data, "pulse_ms", PAINT_PULSE_MS_DEFAULT, 10, 2000)
    with lock:
        rec = trigger_paint("manual", pulse)
    return jsonify({"ok": True, "record": rec, "paint_count": paint_count})


@app.post("/api/sweep")
def api_sweep():
    global sweep_enabled, sweep_painted_names, sweep_painted_ids
    global sweep_last_advance_ts, sweep_scan_start_ts, mode
    data = request.get_json(force=True, silent=True) or {}
    with lock:
        if "enabled" in data:
            sweep_enabled = bool(data["enabled"])
            if sweep_enabled and mode != "auto":
                mode = "auto"
            if not sweep_enabled:
                sweep_scan_start_ts = 0.0
        if data.get("reset_painted"):
            sweep_painted_names = set()
            sweep_painted_ids = set()
            sweep_last_advance_ts = 0.0
            sweep_scan_start_ts = 0.0
            _save_painted_memory()
    return jsonify({"ok": True, "sweep_enabled": sweep_enabled,
                    "painted_ids": sorted(sweep_painted_ids),
                    "mode": mode})


@app.post("/api/paint_auto")
def api_paint_auto():
    global paint_auto, paint_auto_min_centered
    data = request.get_json(force=True, silent=True) or {}
    if "enabled" in data:
        paint_auto = bool(data["enabled"])
    if "min_centered" in data:
        paint_auto_min_centered = max(5, min(600, int(data["min_centered"])))
    return jsonify({"ok": True, "paint_auto": paint_auto,
                    "min_centered": paint_auto_min_centered})


@app.post("/api/moving_target")
def api_moving_target():
    global target_moving
    data = request.get_json(force=True, silent=True) or {}
    if "enabled" in data:
        target_moving = bool(data["enabled"])
    return jsonify({"ok": True, "moving": target_moving})


@app.post("/api/bus_moving")
def api_bus_moving():
    global bus_moving
    data = request.get_json(force=True, silent=True) or {}
    if "enabled" in data:
        bus_moving = bool(data["enabled"])
    return jsonify({"ok": True, "bus_moving": bus_moving})


@app.post("/api/mantis_moving")
def api_mantis_moving():
    global mantis_moving
    data = request.get_json(force=True, silent=True) or {}
    if "enabled" in data:
        mantis_moving = bool(data["enabled"])
    return jsonify({"ok": True, "mantis_moving": mantis_moving})


@app.post("/api/mantis_speed")
def api_mantis_speed():
    global mantis_drive_speed
    data = request.get_json(force=True, silent=True) or {}
    s = float(data.get("speed", mantis_drive_speed))
    mantis_drive_speed = max(0.5, min(MANTIS_DRIVE_MAX_SPEED, s))
    return jsonify({"ok": True, "speed": mantis_drive_speed})


@app.post("/api/mantis_drive")
def api_mantis_drive():
    """V-key drive. Accepts velocity setpoints (preferred) or one-shot
    deltas:
       {"vx": 4.0, "vy": 0, "vyaw": 0}   # smooth car-like drive at 4 m/s
       {"vx": 0,   "vy": 0, "vyaw": 0}   # stop
       {"reset": true}                   # snap back to spawn (0,-8)
    Velocities are in BODY frame: vx = forward, vy = strafe-left.
    Disables the auto-sway oscillation so user input wins."""
    global mantis_chassis_x, mantis_chassis_y, mantis_chassis_yaw
    global mantis_drive_vx, mantis_drive_vy, mantis_drive_vyaw, mantis_moving
    data = request.get_json(force=True, silent=True) or {}
    if data.get("reset"):
        import math as _m, subprocess as _sp
        mantis_chassis_x, mantis_chassis_y, mantis_chassis_yaw = 0.0, -8.0, 0.0
        mantis_drive_vx = mantis_drive_vy = mantis_drive_vyaw = 0.0
        req = ('name: "mantis_robot" '
               'position { x: 0.0 y: -8.0 z: 0.0 } '
               'orientation { w: 1.0 }')
        try:
            _sp.run(
                ["gz", "service", "-s", "/world/mantis_robot_world/set_pose",
                 "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
                 "--timeout", "300", "--req", req],
                capture_output=True, timeout=1.0, check=False,
            )
        except Exception:
            pass
    if "vx" in data:
        mantis_drive_vx = clamp(float(data["vx"]),
                                -MANTIS_DRIVE_MAX_SPEED, MANTIS_DRIVE_MAX_SPEED)
    if "vy" in data:
        mantis_drive_vy = clamp(float(data["vy"]),
                                -MANTIS_DRIVE_MAX_SPEED, MANTIS_DRIVE_MAX_SPEED)
    if "vyaw" in data:
        mantis_drive_vyaw = clamp(float(data["vyaw"]),
                                  -MANTIS_DRIVE_MAX_YAW, MANTIS_DRIVE_MAX_YAW)
    mantis_moving = False  # manual drive overrides auto sway
    return jsonify({"ok": True,
                    "x": mantis_chassis_x, "y": mantis_chassis_y,
                    "yaw": mantis_chassis_yaw,
                    "vx": mantis_drive_vx, "vy": mantis_drive_vy,
                    "vyaw": mantis_drive_vyaw})


@app.post("/api/cars_moving")
def api_cars_moving():
    global cars_moving
    data = request.get_json(force=True, silent=True) or {}
    if "enabled" in data:
        cars_moving = bool(data["enabled"])
    return jsonify({"ok": True, "cars_moving": cars_moving})


@app.post("/api/people_walking")
def api_people_walking():
    global people_walking
    data = request.get_json(force=True, silent=True) or {}
    if "enabled" in data:
        people_walking = bool(data["enabled"])
    return jsonify({"ok": True, "people_walking": people_walking})


def _apply_zoom_locked(new_z: float) -> None:
    """Apply a zoom change. Caller must hold `lock`. Re-maps every pixel
    coord the controller is holding so the same world ray maps to the new
    zoomed-frame pixel — otherwise the anchor refers to the old crop and
    the controller slews off the target."""
    global zoom_factor, selected_anchor_xy, smoothed_cx, smoothed_cy
    global target_vx_pix_s, target_vy_pix_s
    global pan_i_deg, tilt_i_deg, last_ex_norm, last_ey_norm, zoom_changed_ts
    global last_target_w, last_target_h, pan_deg, tilt_deg
    old_z = max(0.01, zoom_factor)
    ratio = new_z / old_z
    cx_img = IMG_W / 2.0
    cy_img = IMG_H / 2.0
    if selected_anchor_xy is not None:
        ax, ay = selected_anchor_xy
        selected_anchor_xy = (cx_img + (ax - cx_img) * ratio,
                              cy_img + (ay - cy_img) * ratio)
    if smoothed_init:
        smoothed_cx = cx_img + (smoothed_cx - cx_img) * ratio
        smoothed_cy = cy_img + (smoothed_cy - cy_img) * ratio
    target_vx_pix_s *= ratio
    target_vy_pix_s *= ratio
    last_target_w *= ratio
    last_target_h *= ratio
    pan_i_deg = 0.0
    tilt_i_deg = 0.0
    last_ex_norm = 0.0
    last_ey_norm = 0.0
    if joint_state_stamp:
        pan_deg = actual_pan_deg
        tilt_deg = actual_tilt_deg
    zoom_factor = new_z
    if abs(new_z - old_z) > 0.01:
        zoom_changed_ts = time.time()


@app.post("/api/zoom")
def api_zoom():
    data = request.get_json(force=True, silent=True) or {}
    new_z = safe_float(data, "zoom", zoom_factor, ZOOM_MIN, ZOOM_MAX)
    with lock:
        _apply_zoom_locked(new_z)
    return jsonify({"ok": True, "zoom": zoom_factor})


@app.post("/api/auto_zoom")
def api_auto_zoom():
    global auto_zoom_enabled
    data = request.get_json(force=True, silent=True) or {}
    if "enabled" in data:
        auto_zoom_enabled = bool(data["enabled"])
    return jsonify({"ok": True, "auto_zoom": auto_zoom_enabled})


@app.get("/api/channels")
def api_channels_get():
    return jsonify({
        "channels": dict(paint_channels),
        "udp": {"host": paint_udp_addr[0], "port": paint_udp_addr[1]},
        "tcp": {"host": paint_tcp_addr[0], "port": paint_tcp_addr[1]},
        "serial": {"port": paint_serial_port, "baud": paint_serial_baud},
    })


@app.post("/api/channels")
def api_channels_set():
    global paint_udp_addr, paint_tcp_addr, paint_serial_port, paint_serial_baud
    data = request.get_json(force=True, silent=True) or {}
    with lock:
        if "channels" in data and isinstance(data["channels"], dict):
            for k, v in data["channels"].items():
                if k in paint_channels:
                    paint_channels[k] = bool(v)
        if "udp" in data:
            paint_udp_addr = (
                str(data["udp"].get("host", paint_udp_addr[0])),
                safe_int(data["udp"], "port", paint_udp_addr[1], 1, 65535),
            )
        if "tcp" in data:
            paint_tcp_addr = (
                str(data["tcp"].get("host", paint_tcp_addr[0])),
                safe_int(data["tcp"], "port", paint_tcp_addr[1], 1, 65535),
            )
        if "serial" in data:
            paint_serial_port = str(data["serial"].get("port", paint_serial_port))
            paint_serial_baud = safe_int(data["serial"], "baud",
                                         paint_serial_baud, 300, 4_000_000)
    return jsonify({"ok": True, "channels": dict(paint_channels)})


def _ollama_list_models() -> list[str]:
    try:
        import urllib.request as _ur
        r = _ur.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2)
        import json as _json
        data = _json.loads(r.read())
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:
        return []


def _agent_pick_default():
    global agent_model
    if agent_model:
        return agent_model
    models = _ollama_list_models()
    if not models:
        return None
    # Prefer small, fast models for tool-call use.
    pref = ("llama3.2", "llama3.1", "qwen2.5", "phi3", "mistral", "gemma")
    for p in pref:
        for m in models:
            if p in m.lower():
                agent_model = m
                return m
    agent_model = models[0]
    return agent_model


def _agent_execute(action: str, payload: dict | None = None) -> str:
    global mode, sweep_enabled, paint_auto
    payload = payload or {}
    a = action.lower()
    with lock:
        if a == "paint":
            rec = trigger_paint("agent", PAINT_PULSE_MS_DEFAULT)
            return f"painted #{rec['n']}"
        if a == "home":
            mode = "home"; return "mode=home"
        if a == "stop":
            mode = "stop"; clear_selection(); return "mode=stop"
        if a in ("track_on", "tracking_on"):
            mode = "auto"; return "tracking on"
        if a in ("track_off", "tracking_off"):
            mode = "manual"; return "tracking off"
        if a in ("sweep_on", "serial_on"):
            sweep_enabled = True; return "auto serial tracker on"
        if a in ("sweep_off", "serial_off"):
            sweep_enabled = False; return "auto serial tracker off"
        if a == "auto_paint_on":
            paint_auto = True; return "auto paint on"
        if a == "auto_paint_off":
            paint_auto = False; return "auto paint off"
        if a == "clear":
            clear_selection(); return "selection cleared"
        if a in ("select_name", "select"):
            name = str(payload.get("name", "")).strip()
            if not name:
                return "select_name needs a target name"
            for d in (detections or recent_detections):
                if d.name.lower() == name.lower():
                    global selected_id, selected_name, selected_anchor_xy
                    selected_id = d.det_id
                    selected_name = d.name
                    x1, y1, x2, y2 = d.bbox
                    selected_anchor_xy = ((x1 + x2) / 2, (y1 + y2) / 2)
                    mode = "auto"
                    return f"selected {d.name}"
            return f"no detection matches '{name}'"
    return f"unknown action '{action}'"


ACTION_INTENT_WORDS = {
    "paint": {"paint", "shoot", "fire", "mark", "spray", "trigger", "tag"},
    "home": {"home", "reset", "center", "neutral", "default", "park"},
    "stop": {"stop", "halt", "freeze", "pause", "abort", "hold"},
    "track_on": {"track", "tracking", "follow", "chase", "lock"},
    "tracking_on": {"track", "tracking", "follow", "chase", "lock"},
    "track_off": {"stop tracking", "untrack", "release", "off"},
    "tracking_off": {"stop tracking", "untrack", "release", "off"},
    "sweep_on": {"sweep", "serial", "all", "scan", "every"},
    "sweep_off": {"sweep", "off", "stop sweep", "stop scan"},
    "serial_on": {"sweep", "serial", "all", "scan"},
    "serial_off": {"stop", "off"},
    "auto_paint_on": {"auto", "automatic", "paint"},
    "auto_paint_off": {"manual", "stop auto", "off"},
    "select_name": {"select", "pick", "target", "aim", "choose", "switch"},
    "select": {"select", "pick", "target", "aim", "choose", "switch"},
    "clear": {"clear", "drop", "forget", "deselect", "unselect", "remove"},
}


def _user_wants_action(user_text: str, action: str) -> bool:
    t = user_text.lower()
    # Trivial greetings
    if t.strip() in {"hi", "hello", "hey", "yo", "ok", "okay", "thanks", "thx",
                     "thank you", "bye", "?"}:
        return False
    # Questions usually want info, not action — unless they explicitly include
    # an imperative word.
    keywords = ACTION_INTENT_WORDS.get(action.lower(), set())
    has_kw = any(k in t for k in keywords)
    # Allow action if user used one of its intent words.
    return has_kw


VALID_ACTIONS = {
    "paint", "home", "stop", "track_on", "track_off", "tracking_on",
    "tracking_off", "sweep_on", "sweep_off", "serial_on", "serial_off",
    "auto_paint_on", "auto_paint_off", "select_name", "select", "clear",
}


def _agent_parse(reply: str) -> tuple[str | None, dict]:
    """Only return an action if the reply explicitly contains a JSON
    {action: ...} block. No keyword fallback — that was firing actions on
    every greeting that happened to contain 'home' or 'paint' in the
    model's chit-chat."""
    import json as _json, re as _re
    matches = _re.findall(r"\{[^{}]*\"action\"[^{}]*\}", reply)
    for m in matches:
        try:
            obj = _json.loads(m)
        except Exception:
            continue
        if not isinstance(obj, dict) or "action" not in obj:
            continue
        action = str(obj.pop("action")).strip().lower()
        if action not in VALID_ACTIONS:
            continue
        return action, obj
    return None, {}


@app.get("/api/agent/models")
def api_agent_models():
    return jsonify({"models": _ollama_list_models(),
                    "selected": agent_model,
                    "enabled": agent_enabled,
                    "status": agent_status,
                    "url": OLLAMA_URL})


@app.post("/api/agent/enable")
def api_agent_enable():
    global agent_enabled, agent_model, agent_status
    data = request.get_json(force=True, silent=True) or {}
    with lock:
        if "enabled" in data:
            agent_enabled = bool(data["enabled"])
        if data.get("model"):
            agent_model = str(data["model"])
        if agent_enabled and agent_model is None:
            _agent_pick_default()
        agent_status = "ready" if agent_enabled and agent_model else "idle"
    return jsonify({"ok": True, "enabled": agent_enabled,
                    "model": agent_model, "status": agent_status})


@app.post("/api/agent/chat")
def api_agent_chat():
    global agent_status
    data = request.get_json(force=True, silent=True) or {}
    text = str(data.get("message", "")).strip()
    if not text:
        return jsonify({"ok": False, "error": "empty message"})
    if not agent_enabled:
        return jsonify({"ok": False, "error": "agent disabled"})
    model = agent_model or _agent_pick_default()
    if not model:
        return jsonify({"ok": False, "error": "no ollama model available"})

    sys_prompt = (
        "You are MANTIS, a helpful assistant. You can talk freely about any "
        "topic AND you can optionally control a pan/tilt camera in a Gazebo "
        "simulation. \n\n"
        "Tool use rule: ONLY emit a JSON action when the user CLEARLY requests "
        "an action on the MANTIS system (e.g. 'paint the car', 'start "
        "tracking', 'stop', 'select the truck'). For greetings, questions, "
        "explanations, chit-chat, or anything else, reply in plain text only — "
        "no JSON. \n\n"
        "When you DO act, put the JSON on its own line as the very last line, "
        "preceded by a 1-sentence confirmation. Format examples:\n"
        "  Got it, painting now.\n"
        "  {\"action\": \"paint\"}\n"
        "  Switching to the car.\n"
        "  {\"action\": \"select_name\", \"name\": \"car\"}\n\n"
        "Allowed actions: paint, home, stop, track_on, track_off, sweep_on, "
        "sweep_off, auto_paint_on, auto_paint_off, select_name (with name), "
        "clear. \n\n"
        "Do NOT invent actions or arguments. If the user's request is "
        "ambiguous, ask a clarifying question in plain text instead of "
        "guessing an action."
    )
    det_names = sorted({d.name for d in (detections or recent_detections)})
    user_ctx = (
        f"User message: {text}\n"
        f"Current mode: {mode}. Tracking on: {mode == 'auto'}. "
        f"Sweep: {sweep_enabled}. AutoPaint: {paint_auto}. "
        f"Pan: {actual_pan_deg:.1f} deg. Tilt: {actual_tilt_deg:.1f} deg. "
        f"Detected: {', '.join(det_names) or 'none'}."
    )
    try:
        import urllib.request as _ur, json as _json
        body = _json.dumps({
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_ctx},
            ],
        }).encode()
        req = _ur.Request(f"{OLLAMA_URL}/api/chat", data=body,
                          headers={"Content-Type": "application/json"})
        agent_status = "thinking"
        r = _ur.urlopen(req, timeout=30)
        agent_status = "ready"
        resp = _json.loads(r.read())
        reply = resp.get("message", {}).get("content", "")
    except Exception as exc:
        agent_status = f"err: {exc}"
        return jsonify({"ok": False, "error": str(exc)})

    action, payload_args = _agent_parse(reply)
    action_result = ""
    # Sanity check: don't execute an action if the USER message doesn't look
    # like an action request. Small models love to invent JSON tool calls in
    # response to greetings or factual questions.
    if action and not _user_wants_action(text, action):
        action_result = "(skipped — message looked conversational)"
        action = None
    if action:
        action_result = _agent_execute(action, payload_args)
    entry = {"time": round(time.time(), 3), "user": text, "reply": reply,
             "action": action, "action_result": action_result}
    with lock:
        agent_chat_log.insert(0, entry)
        del agent_chat_log[64:]
    return jsonify({"ok": True, "reply": reply, "action": action,
                    "action_result": action_result, "model": model})


@app.get("/api/agent/log")
def api_agent_log():
    return jsonify({"log": agent_chat_log[:32]})


@app.post("/api/click_target")
def api_click_target():
    global mode, jog_pan_target, jog_tilt_target
    data = request.get_json(force=True, silent=True) or {}
    x = safe_float(data, "x", IMG_W / 2.0, 0.0, float(IMG_W))
    y = safe_float(data, "y", IMG_H / 2.0, 0.0, float(IMG_H))
    nx = (x - IMG_W / 2.0) / (IMG_W / 2.0)
    ny = (y - IMG_H / 2.0) / (IMG_H / 2.0)
    pan_off = math.degrees(math.atan(math.tan(HFOV_RAD / 2.0) * nx))
    tilt_off = math.degrees(math.atan(math.tan(VFOV_RAD / 2.0) * ny))
    with lock:
        clear_selection()
        base_pan = actual_pan_deg if joint_state_stamp else pan_deg
        base_tilt = actual_tilt_deg if joint_state_stamp else tilt_deg
        jog_pan_target = clamp(base_pan + PAN_SIGN * pan_off,
                               PAN_LIMIT[0], PAN_LIMIT[1])
        jog_tilt_target = clamp(base_tilt + TILT_SIGN * tilt_off,
                                TILT_LIMIT[0], TILT_LIMIT[1])
        mode = "manual"
    return jsonify({"ok": True, "mode": mode,
                    "pan_target": jog_pan_target,
                    "tilt_target": jog_tilt_target,
                    "pan_off_deg": pan_off, "tilt_off_deg": tilt_off})


@app.post("/api/stop")
def api_stop():
    global mode, last_command_pan_deg, last_command_tilt_deg
    with lock:
        clear_selection()
        last_command_pan_deg = pan_deg
        last_command_tilt_deg = tilt_deg
        mode = "stop"
    return jsonify({"ok": True, "mode": mode,
                    "held_pan_deg": last_command_pan_deg,
                    "held_tilt_deg": last_command_tilt_deg})


@app.post("/api/detector")
def api_detector():
    global detector_mode, detections
    data = request.get_json(force=True, silent=True) or {}
    requested = str(data.get("mode", "")).lower()
    if requested not in ("auto", "color", "world"):
        return jsonify({"ok": False, "message": "detector mode must be auto|color|world"}), 400
    with lock:
        if detector_mode != requested:
            detections = []
        detector_mode = requested
    return jsonify({"ok": True, "detector_mode": detector_mode,
                    "yolo_status": yolo_status,
                    "tracker": ("ByteTrack" if detector_mode in ("auto", "world")
                                else "AnchorMatch")})


def _gz_camera_alive() -> bool:
    """True if /mantis/nose_camera/image has produced a frame within the
    last second. Used by auto-detect to know if Gazebo is up."""
    return latest_stamp > 0 and (time.time() - latest_stamp) < 1.5


def _probe_webcam(idx: int) -> bool:
    """Open camera idx briefly to check if it responds with a frame."""
    try:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            return False
        ok, frame = cap.read()
        cap.release()
        return bool(ok and frame is not None)
    except Exception:
        return False


def _auto_detect_source() -> tuple[str, str]:
    """Return (source_string, message). Prefers Gazebo + Mantis joints, then
    falls back to webcam 0, 1, 2."""
    if _gz_camera_alive() and joint_state_stamp > 0:
        return "gz", "Gazebo + Mantis joints live"
    if _gz_camera_alive():
        return "gz", "Gazebo camera live (no joint state)"
    for idx in (0, 1, 2):
        if _probe_webcam(idx):
            return f"webcam:{idx}", f"webcam #{idx} responds"
    return "", "no source responded (gz topic silent, no webcam)"


@app.post("/api/source")
def api_source():
    """Switch the live frame source. Accepts:
      {"source": "gz"}
      {"source": "webcam:<index>"}
      {"source": "auto"}     — probes gz first, then webcam 0/1/2.
    Gazebo stays connected via gz_transport in the background; the webcam
    thread takes priority by overwriting latest_raw."""
    global frame_source
    data = request.get_json(force=True, silent=True) or {}
    src = str(data.get("source", "")).lower().strip()
    detected_msg = ""
    if src == "auto":
        src, detected_msg = _auto_detect_source()
        if not src:
            return jsonify({"ok": False, "error": detected_msg}), 400
    if src == "gz":
        _stop_webcam()
        frame_source = "gz"
        return jsonify({"ok": True, "source": frame_source,
                        "detected": detected_msg})
    if src.startswith("webcam:"):
        try:
            idx = int(src.split(":", 1)[1])
        except Exception:
            return jsonify({"ok": False, "error": "bad webcam index"}), 400
        _start_webcam(idx)
        time.sleep(0.5)
        if _webcam_last_error:
            return jsonify({"ok": False, "error": _webcam_last_error}), 400
        frame_source = src
        return jsonify({"ok": True, "source": frame_source,
                        "detected": detected_msg})
    return jsonify({"ok": False, "error": "source must be auto|gz|webcam:N"}), 400


@app.post("/api/world_classes")
def api_world_classes():
    """Update the open-vocab class prompts used by YOLO-World. Accepts
    {"classes": ["red truck", "person", ...]}. Empty list keeps current."""
    global _yoloworld_classes
    data = request.get_json(force=True, silent=True) or {}
    raw = data.get("classes")
    if isinstance(raw, str):
        raw = [s.strip() for s in raw.split(",") if s.strip()]
    if isinstance(raw, list) and raw:
        with lock:
            _yoloworld_classes = [str(x).strip() for x in raw if str(x).strip()]
    return jsonify({"ok": True, "classes": list(_yoloworld_classes)})


@app.get("/api/world_classes")
def api_world_classes_get():
    return jsonify({"classes": list(_yoloworld_classes)})


@app.post("/api/command")
def api_command_compat():
    return jsonify({"ok": False,
                    "message": "reload page; live camera UI uses /api/select"})


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true",
                    help="run control loop only, do not serve Web UI")
    ap.add_argument("--auto", action="store_true",
                    help="start with Auto Serial Tracker enabled "
                    "(autonomous painting on boot)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5055)
    args = ap.parse_args()
    if args.auto:
        sweep_enabled = True
        paint_auto = True
        mode = "auto"
    if args.headless:
        print("[mantis] headless mode — control loop only, no Web UI",
              flush=True)
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
    else:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
