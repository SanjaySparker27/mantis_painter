#!/usr/bin/env python3
"""Comprehensive multi-trial bench. Sweeps over yaw rates + chassis-drive
combos, records per-trial centering / drift / lock stability, then writes
a summary table + a matplotlib graph (docs/assets/bench_progress.png).
"""
from __future__ import annotations
import json, statistics, sys, time, urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WEB = "http://127.0.0.1:5055"
OUT_PNG = Path("/home/sanju/MANTIS_PAINTER/docs/assets/bench_progress.png")


def P(p, b):
    urllib.request.urlopen(
        urllib.request.Request(
            f"{WEB}{p}", data=json.dumps(b).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        ),
        timeout=15,
    ).read()


def G(p):
    return json.loads(urllib.request.urlopen(f"{WEB}{p}", timeout=8).read())


def lock_target():
    P("/api/mantis_drive", {"reset": True})
    time.sleep(2)
    P("/api/mode", {"mode": "home"})
    time.sleep(3)
    P("/api/mode", {"mode": "manual"})
    P("/api/jog", {"dpan": 22, "dtilt": -5})
    time.sleep(5)
    d = G("/api/status")
    if not d["detections"]:
        return None
    t = max(d["detections"],
            key=lambda x: (x["bbox"][2] - x["bbox"][0])
                          * (x["bbox"][3] - x["bbox"][1]))
    cx = (t["bbox"][0] + t["bbox"][2]) / 2
    cy = (t["bbox"][1] + t["bbox"][3]) / 2
    P("/api/select", {"x": cx, "y": cy})
    time.sleep(4)
    return G("/api/status")["selected_id"]


def trial(label, vyaw, vx, dur):
    sel0 = lock_target()
    if sel0 is None:
        return {"label": label, "vyaw": vyaw, "vx": vx,
                "ok": 0, "ex_max": None, "drift": None,
                "ids": [], "pass": False}
    tw0 = G("/api/status")["target_world_pan_deg"]
    P("/api/mantis_drive", {"vyaw": vyaw, "vx": vx})
    s = []
    ids = []
    drift = []
    t0 = time.time()
    direction = 1
    last_switch = t0
    while time.time() - t0 < dur:
        # alternate direction every 2 s so test stays in pan range
        if time.time() - last_switch >= 2.0:
            direction = -direction
            P("/api/mantis_drive",
              {"vyaw": vyaw * direction, "vx": vx * direction})
            last_switch = time.time()
        d = G("/api/status")
        chassis = d.get("mantis_chassis_yaw_deg", 0.0)
        pan = d["pan_deg"]
        cam_world = chassis + pan
        drift.append(cam_world - (tw0 or cam_world))
        sel = d["selected_id"]
        if sel is not None and (not ids or ids[-1] != sel):
            ids.append(sel)
        locked = next((x for x in d["detections"] if x["id"] == sel), None)
        if locked:
            cx = (locked["bbox"][0] + locked["bbox"][2]) / 2
            s.append((cx - 640) / 640)
        time.sleep(0.1)
    P("/api/mantis_drive", {"vyaw": 0, "vx": 0})
    time.sleep(1)
    expected = int(dur * 10)
    ex_max = max(abs(x) for x in s) if s else None
    drift_max = max(abs(w) for w in drift) if drift else 0
    real_pct = len(s) / expected if expected else 0
    passed = bool(ex_max is not None
                  and ex_max < 0.25 and real_pct >= 0.30)
    return {"label": label, "vyaw": vyaw, "vx": vx,
            "ok": len(s), "expected": expected,
            "real_pct": real_pct,
            "ex_max": ex_max,
            "ex_mean": statistics.mean(s) if s else None,
            "ex_std": statistics.stdev(s) if len(s) > 1 else 0.0,
            "drift_max": drift_max,
            "ids": ids, "pass": passed}


def main():
    scenarios = []
    # Yaw-only sweep
    for v in [-0.7, -0.5, -0.3, -0.15, 0.15, 0.3, 0.5, 0.7]:
        scenarios.append((f"yaw{v:+.2f}", v, 0.0, 5.0))
    # Drive forward / back with mild yaw
    for vx in [4.0, 8.0]:
        for vyaw in [0.0, 0.3]:
            scenarios.append((f"vx{vx:.1f}_yaw{vyaw:.1f}", vyaw, vx, 5.0))

    print(f"FULL BENCH @ {time.strftime('%H:%M:%S')}")
    print(f"scenarios: {len(scenarios)}", flush=True)
    results = []
    for i, sc in enumerate(scenarios):
        label, vyaw, vx, dur = sc
        print(f"\n[{i+1:2d}/{len(scenarios)}] {label}", flush=True)
        r = trial(label, vyaw, vx, dur)
        tag = "PASS" if r["pass"] else "FAIL"
        if r["ex_max"] is not None:
            print(f"  {tag}  real={r['real_pct']*100:.0f}% "
                  f"ex_max={r['ex_max']:.3f} drift={r['drift_max']:.1f}° "
                  f"ids={r['ids'][:6]}", flush=True)
        else:
            print(f"  {tag}  NO real_det", flush=True)
        results.append(r)

    # Summary
    n_pass = sum(1 for r in results if r["pass"])
    print(f"\n=== AGGREGATE {n_pass}/{len(results)} pass ===", flush=True)
    for r in results:
        print(f"  {r['label']:20s} pass={r['pass']} "
              f"ex_max={r['ex_max']} drift_max={r['drift_max']:.1f}",
              flush=True)

    # Graph
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    labels = [r["label"] for r in results]
    ex_max = [r["ex_max"] or 1.0 for r in results]
    drift = [(r["drift_max"] or 0) / 360.0 for r in results]   # normalize to 0..1ish
    real = [r["real_pct"] or 0 for r in results]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    x = list(range(len(results)))
    colors = ["#5bd97f" if r["pass"] else "#ff6b6b" for r in results]
    axes[0].bar(x, ex_max, color=colors)
    axes[0].axhline(0.25, color="#888", linestyle="--", label="pass threshold 0.25")
    axes[0].set_ylabel("max |ex| (frame fraction)")
    axes[0].set_title("Per-trial centering error (lower = better)")
    axes[0].set_ylim(0, 1.0); axes[0].legend(loc="upper right")
    axes[1].bar(x, drift, color=colors)
    axes[1].set_ylabel("world drift / 360° (normalized)")
    axes[1].set_title("Per-trial world-bearing drift during yaw")
    axes[2].bar(x, real, color=colors)
    axes[2].axhline(0.30, color="#888", linestyle="--", label="pass threshold 0.30")
    axes[2].set_ylabel("real-det fraction")
    axes[2].set_title("Per-trial detection availability")
    axes[2].set_ylim(0, 1.0); axes[2].legend(loc="upper right")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha="right")
    fig.suptitle(f"Mantis tracking bench — {n_pass}/{len(results)} pass "
                 f"@ {time.strftime('%Y-%m-%d %H:%M')}")
    fig.tight_layout()
    fig.savefig(str(OUT_PNG), dpi=110)
    print(f"\nwrote {OUT_PNG}")

    P("/api/mantis_drive", {"reset": True})
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
