#!/usr/bin/env python3
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import cv2
import gz.transport13 as gz_transport
import numpy as np
from flask import Flask, Response, jsonify, request
from gz.msgs10.double_pb2 import Double
from gz.msgs10.image_pb2 import Image
from gz.msgs10.int32_pb2 import Int32
from gz.msgs10.model_pb2 import Model


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


pan_gains = Gains(kp=0.55, ki=0.20, kd=0.04, max_rate_deg_s=25.0,
                  integral_clamp_deg=8.0, deadband_norm=0.012)
tilt_gains = Gains(kp=0.55, ki=0.20, kd=0.04, max_rate_deg_s=18.0,
                   integral_clamp_deg=6.0, deadband_norm=0.016)


lock = threading.Lock()
node = gz_transport.Node()
pan_pub = node.advertise(PAN_TOPIC, Double)
tilt_pub = node.advertise(TILT_TOPIC, Double)
paint_pub = node.advertise(PAINT_TOPIC, Int32)

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
paint_count = 0
paint_last_ts = 0.0
paint_auto = False
paint_auto_min_centered = 25
paint_overlay_marks: list[dict] = []
PAINT_OVERLAY_TTL_S = 1.6

sweep_enabled = False
sweep_painted_names: set[str] = set()
sweep_last_advance_ts = 0.0
SWEEP_PER_TARGET_TIMEOUT_S = 8.0

LOST_GRACE_S = 0.8

YOLO_WEIGHTS_TRY = [
    "yolo12n.pt",
    "/home/sanju/yolo11n.pt",
    "yolo11n.pt",
    "yolov8n.pt",
]
_yolo_model = None
_yolo_tracker_cfg = "bytetrack.yaml"
_yolo_class_palette = {}


def _load_yolo():
    global _yolo_model, yolo_status
    if _yolo_model is not None:
        return _yolo_model
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
            conf=0.30,
            iou=0.5,
            imgsz=640,
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


def publish_angle(pub, deg: float) -> None:
    msg = Double()
    msg.data = math.radians(deg)
    pub.publish(msg)


def publish_pan_tilt() -> None:
    publish_angle(pan_pub, pan_deg)
    publish_angle(tilt_pub, tilt_deg)


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
    selected_id = None
    selected_name = None
    selected_anchor_xy = None
    smoothed_init = False
    target_vx_pix_s = 0.0
    target_vy_pix_s = 0.0
    last_target_seen_ts = 0.0
    reset_controller_state()


MAX_ANCHOR_REASSOC_PX = 240.0


MIN_TRACK_SCORE = 0.15


def resolve_selected_target() -> Detection | None:
    if selected_name is None and selected_id is None:
        return None
    strong = [d for d in detections if d.score >= MIN_TRACK_SCORE]
    if not strong:
        strong = list(detections)
    if detector_mode == "auto" and selected_id is not None:
        for d in strong:
            if d.det_id == selected_id:
                return d
        return None
    if selected_name is not None:
        named = [d for d in strong if d.name == selected_name]
    else:
        named = list(strong)
    if not named:
        return None
    if selected_anchor_xy is None:
        return named[0] if len(named) == 1 else None
    ax, ay = selected_anchor_xy
    best = min(named, key=lambda d: (
        ((d.bbox[0] + d.bbox[2]) / 2 - ax) ** 2
        + ((d.bbox[1] + d.bbox[3]) / 2 - ay) ** 2
    ))
    bcx = (best.bbox[0] + best.bbox[2]) / 2
    bcy = (best.bbox[1] + best.bbox[3]) / 2
    dist = math.hypot(bcx - ax, bcy - ay)
    if dist > MAX_ANCHOR_REASSOC_PX:
        return None
    return best


def pixel_norm_to_angle_deg(norm: float, fov_rad: float) -> float:
    return math.degrees(math.atan(math.tan(fov_rad / 2.0) * norm))


def step_toward(target_deg: float, current_deg: float, max_step_deg: float) -> float:
    delta = clamp(target_deg - current_deg, -max_step_deg, max_step_deg)
    return current_deg + delta


def auto_control_step(width: int, height: int, dt: float) -> None:
    global pan_deg, tilt_deg, pan_i_deg, tilt_i_deg, last_ex_norm, last_ey_norm
    global centered_frames, selected_id, last_target_ts, selected_anchor_xy
    global last_target_cx, last_target_cy, last_target_seen_ts
    global target_vx_pix_s, target_vy_pix_s

    now = time.time()
    target = resolve_selected_target()

    if target is None:
        pan_i_deg *= 0.85
        tilt_i_deg *= 0.85
        last_ex_norm *= 0.5
        last_ey_norm *= 0.5
        target_vx_pix_s *= 0.5
        target_vy_pix_s *= 0.5
        if selected_name is None and selected_id is None:
            # Sweep mode: when idle, pick the first not-yet-painted detection.
            if sweep_enabled:
                candidate = next(
                    (d for d in detections
                     if d.score >= MIN_TRACK_SCORE
                     and d.name not in sweep_painted_names),
                    None,
                )
                if candidate is None and sweep_painted_names:
                    sweep_painted_names.clear()  # restart cycle
                if candidate is not None:
                    x1, y1, x2, y2 = candidate.bbox
                    selected_id = candidate.det_id
                    selected_name = candidate.name
                    selected_anchor_xy = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    global sweep_last_advance_ts
                    sweep_last_advance_ts = now
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
    selected_id = target.det_id

    x1, y1, x2, y2 = target.bbox
    cx_raw = (x1 + x2) / 2.0
    cy_raw = (y1 + y2) / 2.0
    selected_anchor_xy = (cx_raw, cy_raw)

    # Exponential filter on bbox center to remove detector micro-jitter
    global smoothed_cx, smoothed_cy, smoothed_init
    if not smoothed_init:
        smoothed_cx = cx_raw
        smoothed_cy = cy_raw
        smoothed_init = True
    else:
        beta = 0.20
        smoothed_cx = (1 - beta) * smoothed_cx + beta * cx_raw
        smoothed_cy = (1 - beta) * smoothed_cy + beta * cy_raw
    cx = smoothed_cx
    cy = smoothed_cy

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

    pan_err_deg = pixel_norm_to_angle_deg(ex_eff, HFOV_RAD)
    tilt_err_deg = pixel_norm_to_angle_deg(ey_eff, VFOV_RAD)

    pan_i_deg = clamp(pan_i_deg + pan_err_deg * dt,
                      -pan_gains.integral_clamp_deg, pan_gains.integral_clamp_deg)
    tilt_i_deg = clamp(tilt_i_deg + tilt_err_deg * dt,
                       -tilt_gains.integral_clamp_deg, tilt_gains.integral_clamp_deg)

    last_pan_err_deg = pixel_norm_to_angle_deg(last_ex_norm, HFOV_RAD)
    last_tilt_err_deg = pixel_norm_to_angle_deg(last_ey_norm, VFOV_RAD)
    pan_derr_per_s = (pan_err_deg - last_pan_err_deg) / dt
    tilt_derr_per_s = (tilt_err_deg - last_tilt_err_deg) / dt
    last_ex_norm = nx
    last_ey_norm = ny

    pan_u_deg = (pan_gains.kp * pan_err_deg
                 + pan_gains.ki * pan_i_deg
                 + pan_gains.kd * pan_derr_per_s)
    tilt_u_deg = (tilt_gains.kp * tilt_err_deg
                  + tilt_gains.ki * tilt_i_deg
                  + tilt_gains.kd * tilt_derr_per_s)

    # Visual servoing: desired joint angle = actual joint angle + correction
    # derived from image error. Ki accumulates against any inner-PID bias so
    # the joint ends up exactly where the image error is zero. No cascaded
    # lead window — that drags cmd around with actual during overshoot.
    actual_pan = actual_pan_deg if joint_state_stamp else pan_deg
    actual_tilt = actual_tilt_deg if joint_state_stamp else tilt_deg

    desired_pan = actual_pan + PAN_SIGN * pan_u_deg
    desired_tilt = actual_tilt + TILT_SIGN * tilt_u_deg

    # Rate limit + low-pass filter on outgoing command for smooth motion.
    pan_max_step = pan_gains.max_rate_deg_s * dt
    tilt_max_step = tilt_gains.max_rate_deg_s * dt
    pan_step_raw = clamp(desired_pan - pan_deg, -pan_max_step, pan_max_step)
    tilt_step_raw = clamp(desired_tilt - tilt_deg, -tilt_max_step, tilt_max_step)
    lpf = 0.22

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
    else:
        centered_frames = 0
    if (paint_auto and centered_frames == paint_auto_min_centered
            and time.time() - paint_last_ts > 1.5):
        trigger_paint("auto-center-hold")

    if sweep_enabled:
        # Sweep state machine: paint the locked target then advance.
        if (centered_frames >= paint_auto_min_centered
                and time.time() - paint_last_ts > 1.0
                and target.name not in sweep_painted_names):
            trigger_paint(f"sweep:{target.name}")
            sweep_painted_names.add(target.name)
            clear_selection()
            return
        if (sweep_last_advance_ts
                and now - sweep_last_advance_ts > SWEEP_PER_TARGET_TIMEOUT_S):
            # Took too long — skip and try the next one.
            sweep_painted_names.add(target.name)
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

    if mode == "auto":
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
        is_sel = (det.det_id == selected_id) or (
            selected_name is not None and det.name == selected_name)
        color = (0, 215, 255) if is_sel else det.color
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3 if is_sel else 2)
        cv2.putText(out, f"ID {det.det_id} {det.name}", (x1, max(24, y1 - 8)),
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
    cv2.putText(out,
                f"Kp {pan_gains.kp:.2f}  Ki {pan_gains.ki:.2f}  Kd {pan_gains.kd:.2f}  paint:{paint_count}",
                (28, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 200, 220), 2)
    return out


_detection_thread_busy = False


def run_detector(frame: np.ndarray) -> list[Detection]:
    if detector_mode == "color":
        return detect_colored_targets(frame)
    yolo_dets = detect_with_yolo(frame)
    if yolo_dets is None:
        return detect_colored_targets(frame)
    return yolo_dets


def detection_worker(frame: np.ndarray) -> None:
    global detections, recent_detections, recent_detection_stamp
    global _detection_thread_busy, yolo_status
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
    finally:
        _detection_thread_busy = False


def on_image(msg: Image) -> None:
    global latest_raw, latest_annotated, latest_stamp, frame_count
    global _detection_thread_busy
    frame = image_to_bgr(msg)
    if frame is None:
        return
    with lock:
        latest_raw = frame
        control_tick(frame.shape[1], frame.shape[0])
        annotated = draw_overlay(frame)
        ok, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            latest_annotated = jpg.tobytes()
        latest_stamp = time.time()
        frame_count += 1
    if not _detection_thread_busy:
        _detection_thread_busy = True
        threading.Thread(target=detection_worker, args=(frame.copy(),),
                         daemon=True).start()


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


def camera_thread() -> None:
    node.subscribe(Image, CAMERA_TOPIC, on_image)
    node.subscribe(Model, JOINT_STATE_TOPIC, on_joint_state)
    while True:
        time.sleep(1.0)


threading.Thread(target=camera_thread, daemon=True).start()


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
    #feed{width:100%;aspect-ratio:16/9;display:block;background:#050607;cursor:crosshair}
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
  <h1>MANTIS PAINTER &mdash; Nose Camera Tracking</h1>
  <div class="label">Click target. Pan/tilt centers it. Manual mode + jog enabled.</div>
</header>
<main>
  <div>
    <section>
      <div class="title"><span>Live Nose Camera</span><span id="status">connecting</span></div>
      <img id="feed" src="/video_feed">
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
        <button id="mAuto" class="active" title="auto-track the selected target">Tracking: ON</button>
        <button id="mManual" title="ignore detections, use jog buttons only">Manual</button>
        <button id="mHome" title="return to home pose">Home</button>
        <button id="mStop" style="background:#5a2126;border-color:#a23a3a" title="freeze in place">STOP</button>
        <button id="clear" title="forget the current target">Clear target</button>
        <button id="bPaint" style="background:#1f4d8c;border-color:#3a78c0" title="trigger one paint pulse (key P)">PAINT</button>
        <button id="bSweep" title="auto-cycle through every detected target, paint each once">Auto-paint all: OFF</button>
      </div>

      <div class="sect-head">Detector</div>
      <div class="row">
        <button id="dAuto" class="active">YOLO+ByteTrack</button>
        <button id="dColor">Color</button>
        <button id="bClickAim">Click-to-Aim: OFF</button>
        <span class="label" id="yoloStatus" style="align-self:center">init</span>
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
        <span class="label">Keys: WASD or Arrows, Space=home, C=clear, M=manual, T=auto</span>
      </div>

      <div class="sect-head">Tracking tuning (slide to apply live)</div>
      <div class="gain" title="how aggressively the camera chases the target"><span>Speed</span><input id="gKp" type="range" min="0.05" max="1.5" step="0.01" value="0.55"><span id="gKpV">0.55</span></div>
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
const modeBtns={auto:mAuto,manual:mManual,home:mHome,stop:mStop};

function setMode(m){
  if(m==='stop'){
    fetch('/api/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
  }else{
    fetch('/api/mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:m})});
  }
  for(const k in modeBtns) modeBtns[k].classList.toggle('active',k===m);
  modeBadge.textContent=m;
}
mAuto.onclick=()=>setMode('auto');
mManual.onclick=()=>setMode('manual');
mHome.onclick=()=>setMode('home');
mStop.onclick=()=>setMode('stop');
bPaint.onclick=async ()=>{
  bPaint.disabled=true;
  try{ await fetch('/api/paint',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pulse_ms:120})}); }
  catch(e){}
  setTimeout(()=>bPaint.disabled=false, 250);
};
let sweepOn=false;
bSweep.onclick=async ()=>{
  sweepOn=!sweepOn;
  bSweep.textContent='Auto-paint all: '+(sweepOn?'ON':'OFF');
  bSweep.classList.toggle('active',sweepOn);
  await fetch('/api/sweep',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:sweepOn})});
};
document.addEventListener('keypress',e=>{
  if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT') return;
  if(e.key==='p'||e.key==='P'){ bPaint.click(); }
});

function setDetector(d){
  fetch('/api/detector',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:d})});
  dAuto.classList.toggle('active',d==='auto');
  dColor.classList.toggle('active',d==='color');
}
dAuto.onclick=()=>setDetector('auto');
dColor.onclick=()=>setDetector('color');

let clickAim=false;
bClickAim.onclick=()=>{
  clickAim=!clickAim;
  bClickAim.textContent='Click-to-Aim: '+(clickAim?'ON':'OFF');
  bClickAim.classList.toggle('active',clickAim);
};
feed.addEventListener('click', async e=>{
  const r=feed.getBoundingClientRect();
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

document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT') return;
  const k=e.key.toLowerCase();
  if(k==='arrowleft'||k==='a'){jog('pan-left');e.preventDefault();}
  else if(k==='arrowright'||k==='d'){jog('pan-right');e.preventDefault();}
  else if(k==='arrowup'||k==='w'){jog('tilt-up');e.preventDefault();}
  else if(k==='arrowdown'||k==='s'){jog('tilt-down');e.preventDefault();}
  else if(k===' '){setMode('home');e.preventDefault();}
  else if(k==='c'){clear.click();}
  else if(k==='m'){setMode('manual');}
  else if(k==='t'){setMode('auto');}
  else if(k==='escape'||k==='x'){setMode('stop');e.preventDefault();}
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
    modeBadge.textContent=s.mode;
    for(const k in modeBtns) modeBtns[k].classList.toggle('active',k===s.mode);
    yoloStatus.textContent=s.yolo_status||'';
    dAuto.classList.toggle('active',s.detector_mode==='auto');
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
      bSweep.textContent='Auto-paint all: '+(sweepOn?'ON':'OFF');
      bSweep.classList.toggle('active',sweepOn);
    }
    detRows.innerHTML=s.detections.map(d=>`<tr><td><button onclick="selectDetection(${d.id})">${d.id}</button></td><td>${d.name}</td><td>${(d.score*100).toFixed(0)}%</td></tr>`).join('');
    marks.innerHTML=s.virtual_marks.slice(0,8).map(m=>`<tr><td>${m.target}</td><td>${m.pan_deg}</td><td>${m.tilt_deg}</td></tr>`).join('');
  }catch(e){}
}
async function selectDetection(id){
  await fetch('/api/select_detection',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id})});
}
setInterval(poll,250); poll();
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
        while True:
            with lock:
                jpg = latest_annotated
            if jpg is None:
                blank = np.zeros((720, 1280, 3), np.uint8)
                cv2.putText(blank, "Waiting for /mantis/nose_camera/image",
                            (340, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (240, 240, 240), 2)
                _, encoded = cv2.imencode(".jpg", blank)
                jpg = encoded.tobytes()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(1 / 20)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


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
            "yolo_status": yolo_status,
            "paint_count": paint_count,
            "paint_auto": paint_auto,
            "sweep_enabled": sweep_enabled,
            "sweep_painted": sorted(sweep_painted_names),
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

        x = float(data.get("x", -1))
        y = float(data.get("y", -1))
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
        selected_id = hit.det_id
        selected_name = hit.name
        selected_anchor_xy = ((x1 + x2) / 2, (y1 + y2) / 2)
        mode = "auto"
    return jsonify({"ok": True, "selected_id": selected_id,
                    "selected_name": selected_name, "mode": mode})


@app.post("/api/select_detection")
def api_select_detection():
    global selected_id, selected_name, selected_anchor_xy, mode
    data = request.get_json(force=True, silent=True) or {}
    det_id = int(data.get("id", 0))
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
        selected_id = hit.det_id
        selected_name = hit.name
        x1, y1, x2, y2 = hit.bbox
        selected_anchor_xy = ((x1 + x2) / 2, (y1 + y2) / 2)
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
        dpan = float(data.get("dpan", 0.0))
        dtilt = float(data.get("dtilt", 0.0))
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
            pan_gains.kp = 0.55; pan_gains.ki = 0.20; pan_gains.kd = 0.04
            pan_gains.max_rate_deg_s = 25.0; pan_gains.deadband_norm = 0.012
            tilt_gains.kp = 0.55; tilt_gains.ki = 0.20; tilt_gains.kd = 0.04
            tilt_gains.max_rate_deg_s = 18.0; tilt_gains.deadband_norm = 0.016
            reset_controller_state()
            return jsonify({"ok": True, "reset": True})
        for key in ("kp", "ki", "kd"):
            if key in data:
                v = float(data[key])
                setattr(pan_gains, key, v)
                setattr(tilt_gains, key, v)
        if "max_rate" in data:
            v = float(data["max_rate"])
            pan_gains.max_rate_deg_s = v
            tilt_gains.max_rate_deg_s = max(10.0, v * 0.8)
        if "deadband" in data:
            v = float(data["deadband"])
            pan_gains.deadband_norm = v
            tilt_gains.deadband_norm = min(0.10, v * 1.3)
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
    try:
        msg = Int32()
        msg.data = int(pulse_ms)
        paint_pub.publish(msg)
    except Exception as exc:
        record["topic_err"] = str(exc)
    try:
        with open(PAINT_SIGNAL_FILE, "a") as f:
            f.write(f"{record['time']} {paint_count} {pulse_ms} {actual_pan_deg:.3f} {actual_tilt_deg:.3f} {selected_name or '-'}\n")
    except Exception as exc:
        record["file_err"] = str(exc)
    return record


@app.post("/api/paint")
def api_paint():
    data = request.get_json(force=True, silent=True) or {}
    pulse = int(data.get("pulse_ms", PAINT_PULSE_MS_DEFAULT))
    pulse = max(10, min(2000, pulse))
    with lock:
        rec = trigger_paint("manual", pulse)
    return jsonify({"ok": True, "record": rec, "paint_count": paint_count})


@app.post("/api/sweep")
def api_sweep():
    global sweep_enabled, sweep_painted_names, sweep_last_advance_ts, mode
    data = request.get_json(force=True, silent=True) or {}
    with lock:
        if "enabled" in data:
            sweep_enabled = bool(data["enabled"])
        if data.get("reset_painted") or sweep_enabled:
            sweep_painted_names = set()
            sweep_last_advance_ts = 0.0
        if sweep_enabled:
            clear_selection()
            mode = "auto"
    return jsonify({"ok": True, "sweep_enabled": sweep_enabled,
                    "painted_names": sorted(sweep_painted_names)})


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


@app.post("/api/click_target")
def api_click_target():
    global mode, jog_pan_target, jog_tilt_target
    data = request.get_json(force=True, silent=True) or {}
    x = float(data.get("x", IMG_W / 2.0))
    y = float(data.get("y", IMG_H / 2.0))
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
    global detector_mode
    data = request.get_json(force=True, silent=True) or {}
    requested = str(data.get("mode", "")).lower()
    if requested not in ("auto", "color"):
        return jsonify({"ok": False, "message": "detector mode must be auto|color"}), 400
    with lock:
        detector_mode = requested
    return jsonify({"ok": True, "detector_mode": detector_mode,
                    "yolo_status": yolo_status})


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
    app.run(host="127.0.0.1", port=5055, debug=False, threaded=True)
