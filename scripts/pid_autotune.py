#!/usr/bin/env python3
"""
Step-response auto-tune for the outer pan/tilt PID exposed by web_app.py.

The web_app integrates pixel error into a joint-angle command. This script
treats the joint command -> joint position chain as the plant and identifies a
first-order-plus-deadtime (FOPDT) model from a step response captured on the
Gazebo joint-state topic. Cohen-Coon / Ziegler-Nichols (table B) rules then
produce PID gains, which are pushed back via /api/gains.

It does the following:
1. Switch web_app to manual mode.
2. Jog pan to 0 deg, wait for joint to settle.
3. Issue a step to JOG_STEP_DEG, capture joint angle vs time.
4. Fit FOPDT (K, tau, theta).
5. Compute PID gains (clamped to safe bounds).
6. POST gains to /api/gains.
7. Repeat for tilt.

Run after `gz sim` is up with the new model.sdf (joint state publisher must be
present) and `web_app.py` is up.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass

import requests

try:
    import gz.transport13 as gz_transport
    from gz.msgs10.double_pb2 import Double
    from gz.msgs10.model_pb2 import Model
except Exception as exc:
    print(f"gz python bindings missing: {exc}", file=sys.stderr)
    raise


WEB = "http://127.0.0.1:5055"
JOINT_STATE_TOPIC = "/mantis/joint_states"
PAN_CMD = "/mantis/pan_cmd"
TILT_CMD = "/mantis/tilt_cmd"
JOG_STEP_DEG_PAN = 25.0
JOG_STEP_DEG_TILT = 12.0
SETTLE_S = 2.5
CAPTURE_S = 4.0
SAMPLE_HZ = 60.0


@dataclass
class Sample:
    t: float
    pos_deg: float


def _post(path: str, payload: dict) -> dict:
    r = requests.post(f"{WEB}{path}", json=payload, timeout=2.0)
    r.raise_for_status()
    return r.json()


class JointObserver:
    def __init__(self) -> None:
        self.node = gz_transport.Node()
        self.pan_deg: float | None = None
        self.tilt_deg: float | None = None
        self.last_stamp = 0.0
        ok = self.node.subscribe(Model, JOINT_STATE_TOPIC, self._on_msg)
        if not ok:
            raise RuntimeError(f"could not subscribe {JOINT_STATE_TOPIC}")

    def _on_msg(self, msg: Model) -> None:
        for j in msg.joint:
            if not j.axis1.position:  # protobuf default skip
                continue
            pos_rad = float(j.axis1.position)
            if j.name == "pan_joint":
                self.pan_deg = math.degrees(pos_rad)
            elif j.name == "tilt_joint":
                self.tilt_deg = math.degrees(pos_rad)
        self.last_stamp = time.time()


_pub_cache: dict[str, object] = {}
_pub_node = gz_transport.Node()


def _publish_step(topic: str, deg: float) -> None:
    pub = _pub_cache.get(topic)
    if pub is None:
        pub = _pub_node.advertise(topic, Double)
        _pub_cache[topic] = pub
    msg = Double()
    msg.data = math.radians(deg)
    pub.publish(msg)


def _hold(topic: str, deg: float, secs: float, rate_hz: float = 50.0) -> None:
    dt = 1.0 / rate_hz
    t_end = time.time() + secs
    while time.time() < t_end:
        _publish_step(topic, deg)
        time.sleep(dt)


def capture_step(observer: JointObserver, axis: str, baseline_deg: float,
                 step_to_deg: float) -> list[Sample]:
    topic = PAN_CMD if axis == "pan" else TILT_CMD

    _hold(topic, baseline_deg, SETTLE_S)

    samples: list[Sample] = []
    t0 = time.time()
    dt = 1.0 / SAMPLE_HZ
    while time.time() - t0 < CAPTURE_S:
        _publish_step(topic, step_to_deg)
        pos = observer.pan_deg if axis == "pan" else observer.tilt_deg
        if pos is not None:
            samples.append(Sample(time.time() - t0, pos))
        time.sleep(dt)
    return samples


def fit_fopdt(samples: list[Sample], baseline: float, target: float):
    """Estimate K, tau, theta from a step from baseline -> target."""
    if len(samples) < 8:
        return None
    span = target - baseline
    if abs(span) < 1e-3:
        return None
    final = sum(s.pos_deg for s in samples[-5:]) / 5.0
    K = (final - baseline) / span
    threshold = baseline + span * 0.632
    t63 = None
    threshold_low = baseline + span * 0.05
    t_start = None
    for s in samples:
        if t_start is None and (
            (span > 0 and s.pos_deg >= threshold_low)
            or (span < 0 and s.pos_deg <= threshold_low)
        ):
            t_start = s.t
        if t63 is None and (
            (span > 0 and s.pos_deg >= threshold)
            or (span < 0 and s.pos_deg <= threshold)
        ):
            t63 = s.t
            break
    if t_start is None or t63 is None or t63 <= t_start:
        return None
    theta = max(0.01, t_start)
    tau = max(0.02, t63 - t_start)
    return K, tau, theta


def cohen_coon(K: float, tau: float, theta: float):
    """Cohen-Coon PID rule. Returns (Kp, Ki, Kd) for parallel form."""
    K = K if abs(K) > 1e-3 else 1.0
    r = theta / tau
    Kp = (1.35 / K) * (1.0 + 0.18 * r) / max(0.05, 1.0 - 0.39 * r)
    Ti = theta * (2.5 - 2.0 * r) / max(0.1, 1.0 - 0.39 * r)
    Td = theta * 0.37 / max(0.05, 1.0 - 0.81 * r)
    Ki = Kp / max(0.05, Ti)
    Kd = Kp * Td
    return Kp, Ki, Kd


def safe_clamp(Kp: float, Ki: float, Kd: float):
    Kp = max(0.10, min(1.8, abs(Kp)))
    Ki = max(0.00, min(0.80, abs(Ki)))
    Kd = max(0.00, min(0.30, abs(Kd)))
    return Kp, Ki, Kd


def tune_axis(observer: JointObserver, axis: str):
    print(f"\n=== Tuning {axis} ===")
    _post("/api/mode", {"mode": "passthrough"})
    time.sleep(0.3)

    step_deg = JOG_STEP_DEG_PAN if axis == "pan" else JOG_STEP_DEG_TILT
    baseline = 0.0 if axis == "pan" else 12.0
    samples = capture_step(observer, axis, baseline, baseline + step_deg)
    if len(samples) < 10:
        print(f"  not enough samples ({len(samples)}) -- is joint_states topic publishing?")
        return None
    fit = fit_fopdt(samples, baseline, baseline + step_deg)
    if fit is None:
        print("  FOPDT fit failed; joint did not move enough")
        return None
    K, tau, theta = fit
    print(f"  FOPDT: K={K:.3f}  tau={tau:.3f}s  theta={theta:.3f}s")
    Kp_raw, Ki_raw, Kd_raw = cohen_coon(K, tau, theta)
    Kp, Ki, Kd = safe_clamp(Kp_raw, Ki_raw, Kd_raw)
    print(f"  Cohen-Coon raw:    Kp={Kp_raw:.3f} Ki={Ki_raw:.3f} Kd={Kd_raw:.3f}")
    print(f"  Clamped published: Kp={Kp:.3f} Ki={Ki:.3f} Kd={Kd:.3f}")
    return Kp, Ki, Kd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="POST the tuned gains back to web_app via /api/gains")
    args = ap.parse_args()

    obs = JointObserver()
    print(f"waiting for joint state on {JOINT_STATE_TOPIC} ...")
    t0 = time.time()
    while obs.pan_deg is None or obs.tilt_deg is None:
        if time.time() - t0 > 5.0:
            print("no joint state. Did you restart gz sim with the new model.sdf?")
            sys.exit(2)
        time.sleep(0.1)
    print(f"  pan_deg={obs.pan_deg:.2f} tilt_deg={obs.tilt_deg:.2f}")

    pan_gains = tune_axis(obs, "pan")
    tilt_gains = tune_axis(obs, "tilt")

    if args.apply and pan_gains:
        Kp = (pan_gains[0] + (tilt_gains[0] if tilt_gains else pan_gains[0])) / 2
        Ki = (pan_gains[1] + (tilt_gains[1] if tilt_gains else pan_gains[1])) / 2
        Kd = (pan_gains[2] + (tilt_gains[2] if tilt_gains else pan_gains[2])) / 2
        print(f"\nPOST /api/gains kp={Kp:.3f} ki={Ki:.3f} kd={Kd:.3f}")
        _post("/api/gains", {"kp": Kp, "ki": Ki, "kd": Kd})

    _post("/api/mode", {"mode": "home"})
    print("\nDone. Mode reset to home.")


if __name__ == "__main__":
    main()
