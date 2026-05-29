#!/usr/bin/env python3
"""Realistic multi-scenario bench. Each trial simulates a user holding
keys for a sustained period (no 2 Hz snap reversals), then releasing
and watching the persistence-memory recovery.

Pass criterion per trial:
  - ex_max < 0.20 during steady-state motion (after 1 s settle)
  - lock id stable (<=3 unique ids)
  - real-det fraction >= 0.40

Writes docs/assets/bench_progress.png with per-trial bars."""
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


def run_pattern(label, motion_plan):
    """motion_plan: list of (vyaw, vx, duration_s) tuples executed in
    sequence. Records per-frame ex, world drift, id stability."""
    sel0 = lock_target()
    if sel0 is None:
        return {"label": label, "ok": 0, "ex_max": None,
                "drift_max": 0.0, "ids": [], "pass": False}
    tw0 = G("/api/status")["target_world_pan_deg"]
    s = []
    ids = [sel0]
    drift = []
    t_settle = 1.0   # ignore first 1 s of each motion segment
    for vyaw, vx, dur in motion_plan:
        P("/api/mantis_drive", {"vyaw": vyaw, "vx": vx})
        seg_start = time.time()
        while time.time() - seg_start < dur:
            d = G("/api/status")
            chassis = d.get("mantis_chassis_yaw_deg", 0.0)
            pan = d["pan_deg"]
            cam_world = chassis + pan
            drift.append(cam_world - (tw0 or cam_world))
            sel = d["selected_id"]
            if sel is not None and (not ids or ids[-1] != sel):
                ids.append(sel)
            locked = next((x for x in d["detections"]
                           if x["id"] == sel), None)
            if locked and (time.time() - seg_start) >= t_settle:
                cx = (locked["bbox"][0] + locked["bbox"][2]) / 2
                s.append((cx - 640) / 640)
            time.sleep(0.1)
    # release + recovery window
    P("/api/mantis_drive", {"vyaw": 0, "vx": 0})
    rec_start = time.time()
    while time.time() - rec_start < 2.0:
        d = G("/api/status")
        sel = d["selected_id"]
        if sel is not None and (not ids or ids[-1] != sel):
            ids.append(sel)
        locked = next((x for x in d["detections"] if x["id"] == sel), None)
        if locked:
            cx = (locked["bbox"][0] + locked["bbox"][2]) / 2
            s.append((cx - 640) / 640)
        time.sleep(0.1)
    expected = int(sum(d for _, _, d in motion_plan) * 10 - len(motion_plan) * 10)
    ex_max = max(abs(x) for x in s) if s else None
    drift_max = max(abs(w) for w in drift) if drift else 0
    real_pct = len(s) / max(1, expected)
    unique_ids = len(set(ids))
    passed = bool(ex_max is not None
                  and ex_max < 0.20
                  and unique_ids <= 3
                  and real_pct >= 0.40)
    return {"label": label,
            "ok": len(s),
            "expected": expected,
            "real_pct": real_pct,
            "ex_max": ex_max,
            "ex_mean": statistics.mean(s) if s else None,
            "ex_std": statistics.stdev(s) if len(s) > 1 else 0.0,
            "drift_max": drift_max,
            "ids": ids,
            "unique_ids": unique_ids,
            "pass": passed}


def main():
    scenarios = [
        # Realistic single-direction holds
        ("brief_L",       [( 0.30, 0.0, 2.0)]),
        ("brief_R",       [(-0.30, 0.0, 2.0)]),
        ("sustained_L",   [( 0.30, 0.0, 5.0)]),
        ("sustained_R",   [(-0.30, 0.0, 5.0)]),
        ("slow_sustained_L",[( 0.15, 0.0, 6.0)]),
        ("slow_sustained_R",[(-0.15, 0.0, 6.0)]),
        # Drive forward (no yaw)
        ("drive_fwd_slow",[(0.0, 4.0, 4.0)]),
        ("drive_fwd_fast",[(0.0, 8.0, 4.0)]),
        # Drive + gentle steer
        ("drive_curve_R", [(-0.20, 5.0, 4.0)]),
        ("drive_curve_L", [( 0.20, 5.0, 4.0)]),
        # Multi-step user input: turn, brief pause, turn other way
        ("turn_pause_turn",[(0.30, 0.0, 2.0), (0.0, 0.0, 1.0),
                            (-0.30, 0.0, 2.0)]),
        # Stop-and-go: drive, stop, drive
        ("stop_and_go",   [(0.0, 6.0, 2.0), (0.0, 0.0, 1.0),
                           (0.0, 6.0, 2.0)]),
    ]

    print(f"REALISTIC BENCH @ {time.strftime('%H:%M:%S')}", flush=True)
    print(f"scenarios: {len(scenarios)}", flush=True)
    results = []
    for i, (label, plan) in enumerate(scenarios):
        print(f"\n[{i+1:2d}/{len(scenarios)}] {label}", flush=True)
        r = run_pattern(label, plan)
        tag = "PASS" if r["pass"] else "FAIL"
        if r["ex_max"] is not None:
            print(f"  {tag}  ex_max={r['ex_max']:.3f} "
                  f"real={r['real_pct']*100:.0f}% "
                  f"unique_ids={r['unique_ids']} "
                  f"drift={r['drift_max']:.1f}°", flush=True)
        else:
            print(f"  {tag}  NO real_det", flush=True)
        results.append(r)

    n_pass = sum(1 for r in results if r["pass"])
    print(f"\n=== AGGREGATE {n_pass}/{len(results)} pass ===", flush=True)
    for r in results:
        print(f"  {r['label']:22s} pass={r['pass']} "
              f"ex_max={r['ex_max']} real_pct={r['real_pct']:.2f} "
              f"unique_ids={r['unique_ids']}", flush=True)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    labels = [r["label"] for r in results]
    ex_max = [r["ex_max"] or 1.0 for r in results]
    drift = [(r["drift_max"] or 0) / 90.0 for r in results]
    real = [r["real_pct"] or 0 for r in results]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    x = list(range(len(results)))
    colors = ["#5bd97f" if r["pass"] else "#ff6b6b" for r in results]
    axes[0].bar(x, ex_max, color=colors)
    axes[0].axhline(0.20, color="#888", linestyle="--", label="pass < 0.20")
    axes[0].set_ylabel("max |ex| (frame fraction)")
    axes[0].set_title("Per-trial centering error (lower = better)")
    axes[0].set_ylim(0, 1.0); axes[0].legend(loc="upper right")
    axes[1].bar(x, drift, color=colors)
    axes[1].set_ylabel("world drift / 90° (normalized)")
    axes[1].set_title("Per-trial world-bearing drift during motion")
    axes[2].bar(x, real, color=colors)
    axes[2].axhline(0.40, color="#888", linestyle="--", label="pass >= 0.40")
    axes[2].set_ylabel("real-det fraction")
    axes[2].set_title("Per-trial detection availability")
    axes[2].set_ylim(0, 1.05); axes[2].legend(loc="upper right")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha="right")
    fig.suptitle(f"Mantis realistic bench — {n_pass}/{len(results)} pass "
                 f"@ {time.strftime('%Y-%m-%d %H:%M')}")
    fig.tight_layout()
    fig.savefig(str(OUT_PNG), dpi=110)
    print(f"\nwrote {OUT_PNG}")

    P("/api/mantis_drive", {"reset": True})
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
