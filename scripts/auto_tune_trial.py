#!/usr/bin/env python3
"""Run repeated tracking trials, score them, and iterate on PID gains.

Each trial:
  1. home, manual-jog so a target is in view
  2. click the highest-confidence detection
  3. sample status at 5 Hz for ~8 s
  4. compute: time-to-lock, SS mean error, SS std, max overshoot

Then nudge gains and run again. Final best set is left applied.
"""
from __future__ import annotations
import json
import statistics
import time
import urllib.request

WEB = "http://127.0.0.1:5055"


def post(p, b):
    r = urllib.request.Request(
        f"{WEB}{p}", data=json.dumps(b).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(r, timeout=12).read())


def get(p):
    return json.loads(urllib.request.urlopen(f"{WEB}{p}", timeout=8).read())


def run_trial(label, jog_dpan, jog_dtilt, zoom=1.0):
    post("/api/detector", {"mode": "auto"})
    post("/api/zoom", {"zoom": 1.0})
    post("/api/mode", {"mode": "home"})
    time.sleep(2.5)
    post("/api/mode", {"mode": "manual"})
    post("/api/jog", {"dpan": jog_dpan, "dtilt": jog_dtilt})
    time.sleep(4.0)
    if zoom != 1.0:
        post("/api/zoom", {"zoom": zoom})
        time.sleep(1.5)

    d = get("/api/status")
    if not d["detections"]:
        return {"label": label, "ok": False, "reason": "no detection"}
    target = max(d["detections"], key=lambda x: x["score"])
    cx = (target["bbox"][0] + target["bbox"][2]) / 2
    cy = (target["bbox"][1] + target["bbox"][3]) / 2
    post("/api/select", {"x": cx, "y": cy})

    samples = []
    t0 = time.time()
    while time.time() - t0 < 8.0:
        d = get("/api/status")
        det = next(
            (x for x in d["detections"]
             if x["id"] == d["selected_id"]
             or x["name"] == d["selected_name"]),
            None,
        )
        if det:
            ccx = (det["bbox"][0] + det["bbox"][2]) / 2
            ccy = (det["bbox"][1] + det["bbox"][3]) / 2
            ex = (ccx - 640) / 640
            ey = (ccy - 360) / 360
            samples.append((time.time() - t0, ex, ey))
        time.sleep(0.20)
    if zoom != 1.0:
        post("/api/zoom", {"zoom": 1.0})
    if len(samples) < 6:
        return {"label": label, "ok": False,
                "reason": f"only {len(samples)} samples"}

    # Convergence time — first sustained 3 samples with hypot(ex,ey)<0.06
    converge = None
    for i in range(len(samples) - 2):
        if all((samples[j][1] ** 2 + samples[j][2] ** 2) ** 0.5 < 0.06
               for j in (i, i + 1, i + 2)):
            converge = samples[i][0]
            break

    ss = [s for s in samples if s[0] > 4.0]
    if ss:
        exs = [s[1] for s in ss]
        eys = [s[2] for s in ss]
        ex_mean = statistics.mean(exs)
        ex_std = statistics.stdev(exs) if len(exs) > 1 else 0.0
        ey_mean = statistics.mean(eys)
        ey_std = statistics.stdev(eys) if len(eys) > 1 else 0.0
    else:
        ex_mean = ex_std = ey_mean = ey_std = float("nan")

    diverged = any(abs(s[1]) > 0.6 or abs(s[2]) > 0.6
                   for s in samples if s[0] > 2.0)
    return {
        "label": label, "ok": True, "n": len(samples),
        "converge_s": converge,
        "ex_mean": ex_mean, "ex_std": ex_std,
        "ey_mean": ey_mean, "ey_std": ey_std,
        "diverged": diverged,
        "target": target["name"], "score": target["score"],
    }


def score(result):
    if not result["ok"]:
        return -1e9
    if result["diverged"]:
        return -1e6
    if result["converge_s"] is None:
        return -1e3
    return -(
        2.0 * result["converge_s"]
        + 50.0 * (result["ex_mean"] ** 2 + result["ey_mean"] ** 2)
        + 30.0 * (result["ex_std"] ** 2 + result["ey_std"] ** 2)
    )


def apply_gains(gains: dict):
    post("/api/gains", gains)


def main():
    trials = [
        ("car-left", 28, -2, 1.0),
        ("car-right", -10, -2, 1.0),
        ("zoom-2x", 28, -2, 2.0),
    ]

    candidate_gain_sets = [
        {"kp": 0.50, "ki": 0.18, "kd": 0.14, "max_rate": 35, "deadband": 0.008},
        {"kp": 0.55, "ki": 0.22, "kd": 0.16, "max_rate": 40, "deadband": 0.008},
        {"kp": 0.60, "ki": 0.25, "kd": 0.18, "max_rate": 45, "deadband": 0.007},
        {"kp": 0.45, "ki": 0.15, "kd": 0.12, "max_rate": 30, "deadband": 0.010},
        {"kp": 0.55, "ki": 0.30, "kd": 0.20, "max_rate": 45, "deadband": 0.006},
    ]

    best = None
    for gset in candidate_gain_sets:
        apply_gains(gset)
        time.sleep(0.5)
        print(f"\n=== gains {gset} ===", flush=True)
        results = []
        for label, dp, dt, z in trials:
            r = run_trial(label, dp, dt, z)
            r_repr = (f"  {label}: ok={r.get('ok')} converge={r.get('converge_s')} "
                      f"ex={r.get('ex_mean'):+.4f}±{r.get('ex_std'):.4f} "
                      f"ey={r.get('ey_mean'):+.4f}±{r.get('ey_std'):.4f} "
                      f"div={r.get('diverged')} target={r.get('target')}"
                      if r["ok"] else f"  {label}: FAIL {r.get('reason')}")
            print(r_repr, flush=True)
            results.append(r)
        total = sum(score(r) for r in results)
        print(f"  TOTAL SCORE = {total:+.3f}", flush=True)
        if best is None or total > best[0]:
            best = (total, gset, results)

    print(f"\n=== BEST gains: {best[1]} (score {best[0]:+.3f}) ===")
    apply_gains(best[1])


if __name__ == "__main__":
    main()
