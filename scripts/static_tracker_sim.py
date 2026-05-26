#!/usr/bin/env python3
"""CPU-friendly static pan/tilt vehicle tracking simulation.

This sim intentionally models only sensing, tracking, and pan/tilt control.
The "mark" event is a virtual log entry when a tracked vehicle remains centered.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


W, H = 1280, 720
HFOV = math.radians(58.0)
VFOV = math.radians(34.0)
DT = 1.0 / 30.0
PAN_LIMIT = (math.radians(-85.29999907243065), math.radians(89.19999610737291))
TILT_LIMIT = (math.radians(-39.99999883637168), math.radians(30.000000834826057))


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def wrap_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


@dataclass
class Vehicle:
    name: str
    color: tuple[int, int, int]
    path: list[tuple[float, float]]
    speed: float
    length: float
    width: float
    height: float
    phase: float = 0.0
    distance: float = 0.0

    def pose(self, t: float) -> tuple[float, float, float]:
        points = self.path
        seg_lengths = [
            math.hypot(points[(i + 1) % len(points)][0] - points[i][0],
                       points[(i + 1) % len(points)][1] - points[i][1])
            for i in range(len(points))
        ]
        total = sum(seg_lengths)
        d = (self.phase + self.speed * t) % total
        for i, seg_len in enumerate(seg_lengths):
            if d <= seg_len:
                a = points[i]
                b = points[(i + 1) % len(points)]
                u = d / max(seg_len, 1e-6)
                x = a[0] + (b[0] - a[0]) * u
                y = a[1] + (b[1] - a[1]) * u
                yaw = math.atan2(b[1] - a[1], b[0] - a[0])
                self.distance = math.hypot(x, y)
                return x, y, yaw
            d -= seg_len
        return points[0][0], points[0][1], 0.0


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    score: float
    vehicle_name: str


@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    vehicle_name: str
    score: float
    age: int = 0
    missed: int = 0
    hits: int = 1


class ByteTrackLite:
    """Small IoU tracker shaped like ByteTrack's detection association stage."""

    def __init__(self, iou_threshold: float = 0.25, max_missed: int = 18):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: list[Track] = []

    @staticmethod
    def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        return inter / max(area_a + area_b - inter, 1e-6)

    def update(self, detections: list[Detection]) -> list[Track]:
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_dets = set(range(len(detections)))
        pairs: list[tuple[float, int, int]] = []

        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(detections):
                pairs.append((self.iou(track.bbox, det.bbox), ti, di))
        pairs.sort(reverse=True)

        for score, ti, di in pairs:
            if score < self.iou_threshold or ti not in unmatched_tracks or di not in unmatched_dets:
                continue
            det = detections[di]
            track = self.tracks[ti]
            track.bbox = det.bbox
            track.vehicle_name = det.vehicle_name
            track.score = det.score
            track.missed = 0
            track.hits += 1
            unmatched_tracks.remove(ti)
            unmatched_dets.remove(di)

        for ti in unmatched_tracks:
            self.tracks[ti].missed += 1

        for di in unmatched_dets:
            det = detections[di]
            self.tracks.append(Track(self.next_id, det.bbox, det.vehicle_name, det.score))
            self.next_id += 1

        for track in self.tracks:
            track.age += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return self.tracks


@dataclass
class Turret:
    pan: float = 0.0
    tilt: float = math.radians(-6.0)
    pan_rate: float = 0.0
    tilt_rate: float = 0.0
    last_error_x: float = 0.0
    last_error_y: float = 0.0
    centered_frames: int = 0
    marks: list[tuple[float, int, str]] = field(default_factory=list)

    def update(self, error_x: float, error_y: float, target: Track | None, t: float) -> None:
        max_rate = math.radians(95.0)
        dead = 0.014
        kp_pan = 2.35
        kd_pan = 0.18
        kp_tilt = 1.75
        kd_tilt = 0.12

        if target is None:
            # Acquisition mode: slow sweep until a stable target is detected.
            self.pan_rate = math.radians(16.0) * math.sin(t * 0.55)
            self.tilt_rate *= 0.88
            self.centered_frames = 0
        else:
            ex = 0.0 if abs(error_x) < dead else error_x
            ey = 0.0 if abs(error_y) < dead else error_y
            dx = (ex - self.last_error_x) / DT
            dy = (ey - self.last_error_y) / DT
            commanded_pan = kp_pan * ex + kd_pan * dx
            commanded_tilt = -(kp_tilt * ey + kd_tilt * dy)
            self.pan_rate = 0.72 * self.pan_rate + 0.28 * clamp(commanded_pan, -max_rate, max_rate)
            self.tilt_rate = 0.72 * self.tilt_rate + 0.28 * clamp(commanded_tilt, -max_rate * 0.55, max_rate * 0.55)

            if abs(error_x) < 0.035 and abs(error_y) < 0.045:
                self.centered_frames += 1
            else:
                self.centered_frames = 0

            if self.centered_frames in (18, 90, 180, 270):
                self.marks.append((t, target.track_id, target.vehicle_name))

        self.last_error_x = error_x
        self.last_error_y = error_y
        self.pan = clamp(wrap_pi(self.pan + self.pan_rate * DT), PAN_LIMIT[0], PAN_LIMIT[1])
        self.tilt = clamp(self.tilt + self.tilt_rate * DT, TILT_LIMIT[0], TILT_LIMIT[1])


def project_vehicle(vehicle: Vehicle, pose: tuple[float, float, float], turret: Turret) -> Detection | None:
    x, y, _yaw = pose
    dz = 0.85
    bearing = math.atan2(y, x)
    distance = math.hypot(x, y)
    elev = math.atan2(dz, distance)
    rel_pan = wrap_pi(bearing - turret.pan)
    rel_tilt = elev - turret.tilt

    if abs(rel_pan) > HFOV * 0.72 or abs(rel_tilt) > VFOV * 0.82 or distance < 4.0:
        return None

    cx = W * (0.5 + rel_pan / HFOV)
    cy = H * (0.5 - rel_tilt / VFOV)
    scale = 900.0 / max(distance, 1.0)
    bw = clamp(vehicle.width * scale * 1.8, 24, 260)
    bh = clamp(vehicle.height * scale * 1.9, 18, 170)

    # Deterministic camera noise, small enough to show tracker smoothing needs.
    jitter_x = 4.0 * math.sin(distance * 0.31)
    jitter_y = 3.0 * math.cos(distance * 0.23)
    cx += jitter_x
    cy += jitter_y

    bbox = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
    if bbox[2] < 0 or bbox[0] > W or bbox[3] < 0 or bbox[1] > H:
        return None
    score = clamp(0.98 - distance / 190.0, 0.45, 0.94)
    return Detection(bbox, score, vehicle.name)


def draw_scene(frame: np.ndarray, vehicles: list[Vehicle], poses: dict[str, tuple[float, float, float]], turret: Turret) -> None:
    frame[:] = (178, 190, 196)
    horizon = int(H * 0.47)
    frame[horizon:] = (72, 82, 72)

    vanish = (W // 2, horizon)
    for offset in [-14, -7, 0, 7, 14]:
        x1 = int(W / 2 + offset * 4)
        cv2.line(frame, (x1, H), vanish, (95, 95, 95), 3)
    cv2.rectangle(frame, (0, horizon + 135), (W, horizon + 245), (55, 58, 60), -1)
    for x in range(-120, W + 120, 180):
        cv2.line(frame, (x, horizon + 190), (x + 80, horizon + 190), (230, 220, 80), 4)

    for vehicle in sorted(vehicles, key=lambda v: v.distance, reverse=True):
        det = project_vehicle(vehicle, poses[vehicle.name], turret)
        if det is None:
            continue
        x1, y1, x2, y2 = [int(clamp(v, -50, max(W, H) + 50)) for v in det.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), vehicle.color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (25, 25, 25), 2)
        wheel_y = y2 - max(4, (y2 - y1) // 8)
        cv2.circle(frame, (x1 + max(8, (x2 - x1) // 4), wheel_y), max(4, (x2 - x1) // 10), (20, 20, 20), -1)
        cv2.circle(frame, (x2 - max(8, (x2 - x1) // 4), wheel_y), max(4, (x2 - x1) // 10), (20, 20, 20), -1)


@dataclass
class TargetSelector:
    active_id: int | None = None
    missing_frames: int = 0
    switch_cooldown: int = 0

    def select(self, tracks: list[Track]) -> Track | None:
        visible = [t for t in tracks if t.missed == 0 and t.score > 0.4 and t.hits >= 2]
        if not visible:
            self.missing_frames += 1
            if self.missing_frames > 15:
                self.active_id = None
            self.switch_cooldown = max(0, self.switch_cooldown - 1)
            return None

        if self.active_id is not None:
            for track in visible:
                if track.track_id == self.active_id:
                    self.missing_frames = 0
                    self.switch_cooldown = max(0, self.switch_cooldown - 1)
                    return track

        best = max(visible, key=lambda t: (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1]))
        if self.switch_cooldown == 0:
            self.active_id = best.track_id
            self.switch_cooldown = 20
        self.missing_frames = 0
        return best


def select_target(tracks: list[Track], previous_id: int | None) -> Track | None:
    visible = [t for t in tracks if t.missed == 0 and t.score > 0.4]
    if not visible:
        return None
    if previous_id is not None:
        for track in visible:
            if track.track_id == previous_id:
                return track
    return max(visible, key=lambda t: (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1]))


def run(output_dir: Path, seconds: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    vehicles = [
        Vehicle("prius_like", (55, 65, 210), [(-42, 13), (45, 13), (45, 20), (-42, 20)], 8.0, 4.5, 1.9, 1.5, 0),
        Vehicle("pickup_like", (60, 150, 70), [(-50, -10), (52, -10), (52, -17), (-50, -17)], 6.5, 5.2, 2.1, 1.8, 35),
        Vehicle("van_like", (185, 95, 45), [(25, -38), (25, 38), (34, 38), (34, -38)], 5.2, 5.0, 2.2, 2.2, 20),
    ]
    turret = Turret()
    tracker = ByteTrackLite()
    selector = TargetSelector()

    video_path = output_dir / "static_tracker_demo.mp4"
    csv_path = output_dir / "tracking_metrics.csv"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    rows = []

    for frame_idx in range(int(seconds / DT)):
        t = frame_idx * DT
        poses = {v.name: v.pose(t) for v in vehicles}
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        draw_scene(frame, vehicles, poses, turret)

        detections = [d for v in vehicles if (d := project_vehicle(v, poses[v.name], turret)) is not None]
        tracks = tracker.update(detections)
        target = selector.select(tracks)

        error_x = 0.0
        error_y = 0.0
        if target is not None:
            x1, y1, x2, y2 = target.bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            error_x = (cx - W / 2.0) / (W / 2.0)
            error_y = (cy - H / 2.0) / (H / 2.0)
        turret.update(error_x, error_y, target, t)

        cv2.line(frame, (W // 2 - 34, H // 2), (W // 2 + 34, H // 2), (255, 255, 255), 1)
        cv2.line(frame, (W // 2, H // 2 - 34), (W // 2, H // 2 + 34), (255, 255, 255), 1)
        cv2.rectangle(frame, (W // 2 - 45, H // 2 - 32), (W // 2 + 45, H // 2 + 32), (80, 220, 255), 1)

        for track in tracks:
            if track.missed:
                continue
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            color = (0, 255, 255) if target and track.track_id == target.track_id else (230, 230, 230)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {track.track_id} {track.vehicle_name}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.putText(frame, "MANTIS STL PROJECT - web detection/tracking view",
                    (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2)
        cv2.putText(frame, f"pan {math.degrees(turret.pan):6.1f} deg  tilt {math.degrees(turret.tilt):5.1f} deg  blend limits pan -85.3..89.2 tilt -40..30",
                    (22, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 20, 20), 2)
        cv2.putText(frame, f"target {target.track_id if target else '-'}  marks {len(turret.marks)}",
                    (22, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 20, 20), 2)
        writer.write(frame)

        rows.append({
            "time_s": f"{t:.3f}",
            "target_id": target.track_id if target else "",
            "target_name": target.vehicle_name if target else "",
            "pan_deg": f"{math.degrees(turret.pan):.3f}",
            "tilt_deg": f"{math.degrees(turret.tilt):.3f}",
            "error_x_norm": f"{error_x:.4f}",
            "error_y_norm": f"{error_y:.4f}",
            "visible_tracks": sum(1 for tr in tracks if tr.missed == 0),
            "virtual_marks": len(turret.marks),
            "pan_at_limit": int(abs(turret.pan - PAN_LIMIT[0]) < 1e-4 or abs(turret.pan - PAN_LIMIT[1]) < 1e-4),
            "tilt_at_limit": int(abs(turret.tilt - TILT_LIMIT[0]) < 1e-4 or abs(turret.tilt - TILT_LIMIT[1]) < 1e-4),
            "detections_json": json.dumps([
                {
                    "name": det.vehicle_name,
                    "score": round(det.score, 3),
                    "bbox": [round(v, 1) for v in det.bbox],
                }
                for det in detections
            ], separators=(",", ":")),
            "tracks_json": json.dumps([
                {
                    "id": tr.track_id,
                    "name": tr.vehicle_name,
                    "score": round(tr.score, 3),
                    "missed": tr.missed,
                    "hits": tr.hits,
                    "bbox": [round(v, 1) for v in tr.bbox],
                    "target": bool(target and tr.track_id == target.track_id),
                }
                for tr in tracks if tr.missed == 0
            ], separators=(",", ":")),
        })

    writer.release()
    with csv_path.open("w", newline="") as f:
        fieldnames = list(rows[0].keys())
        out = csv.DictWriter(f, fieldnames=fieldnames)
        out.writeheader()
        out.writerows(rows)

    centered = [
        r for r in rows
        if r["target_id"] and abs(float(r["error_x_norm"])) < 0.05 and abs(float(r["error_y_norm"])) < 0.06
    ]
    print(f"video={video_path}")
    print(f"metrics={csv_path}")
    print(f"frames={len(rows)} duration_s={seconds:.1f}")
    print(f"centered_ratio={len(centered) / max(1, len(rows)):.3f}")
    print(f"virtual_marks={len(turret.marks)}")
    for mark_time, track_id, name in turret.marks[:8]:
        print(f"mark t={mark_time:.2f}s id={track_id} name={name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()
    run(args.output, args.seconds)


if __name__ == "__main__":
    main()
