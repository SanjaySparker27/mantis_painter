#!/usr/bin/env python3
"""Capture a tracking session and produce convergence plot for docs."""
from __future__ import annotations

import json
import sys
import time
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


WEB = "http://127.0.0.1:5055"
OUT_PNG = "/home/sanju/MANTIS_PAINTER/docs/assets/convergence.png"


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{WEB}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


def get(path: str) -> dict:
    return json.loads(urllib.request.urlopen(f"{WEB}{path}", timeout=10).read())


def main():
    post("/api/detector", {"mode": "auto"})
    time.sleep(2)
    post("/api/mode", {"mode": "home"})
    time.sleep(2)
    post("/api/mode", {"mode": "manual"})
    post("/api/jog", {"dpan": 22, "dtilt": -3})
    time.sleep(4)

    d = get("/api/status")
    target = next((x for x in d["detections"] if x["score"] > 0.4), None)
    if target is None and d["detections"]:
        target = d["detections"][0]
    if target is None:
        print("no detections — aborting")
        sys.exit(1)
    cx = (target["bbox"][0] + target["bbox"][2]) / 2
    cy = (target["bbox"][1] + target["bbox"][3]) / 2
    print(f"selecting {target['name']} at ({cx:.0f},{cy:.0f})")
    post("/api/select", {"x": cx, "y": cy})

    t0 = time.time()
    times, pan_cmd, pan_act, tilt_cmd, tilt_act, ex, ey = ([] for _ in range(7))
    for _ in range(60):
        d = get("/api/status")
        det = next(
            (x for x in d["detections"] if x["name"] == d["selected_name"]),
            None,
        )
        t = time.time() - t0
        times.append(t)
        pan_cmd.append(d["pan_deg"])
        pan_act.append(d["actual_pan_deg"])
        tilt_cmd.append(d["tilt_deg"])
        tilt_act.append(d["actual_tilt_deg"])
        if det:
            ccx = (det["bbox"][0] + det["bbox"][2]) / 2
            ccy = (det["bbox"][1] + det["bbox"][3]) / 2
            ex.append((ccx - 640) / 640)
            ey.append((ccy - 360) / 360)
        else:
            ex.append(None)
            ey.append(None)
        time.sleep(0.20)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    a1.plot(times, pan_cmd, label="pan cmd", color="#1f77b4", linewidth=1.6)
    a1.plot(times, pan_act, label="pan actual", color="#1f77b4",
            linestyle="--", linewidth=1.0)
    a1.plot(times, tilt_cmd, label="tilt cmd", color="#d62728", linewidth=1.6)
    a1.plot(times, tilt_act, label="tilt actual", color="#d62728",
            linestyle="--", linewidth=1.0)
    a1.set_ylabel("deg")
    a1.set_title(f"MANTIS tracking convergence — selected {target['name']}")
    a1.legend(loc="best", fontsize=9)
    a1.grid(alpha=0.3)

    a2.plot(times, [e if e is not None else float("nan") for e in ex],
            label="ex (pixel error X, normalized)", color="#2ca02c")
    a2.plot(times, [e if e is not None else float("nan") for e in ey],
            label="ey (pixel error Y, normalized)", color="#ff7f0e")
    a2.axhline(0.0, color="#666", linewidth=0.6)
    a2.axhline(0.030, color="#aaa", linestyle=":", linewidth=0.6, label="deadband")
    a2.axhline(-0.030, color="#aaa", linestyle=":", linewidth=0.6)
    a2.set_xlabel("time (s)")
    a2.set_ylabel("pixel error (normalized)")
    a2.legend(loc="best", fontsize=9)
    a2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=110)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
