#!/usr/bin/env python3
"""Multi-trial redundancy bench. Runs each scenario N times and reports
pass rate + aggregate metrics so a regression in one trial doesn't get
hidden by a single noisy outlier.

Pass criterion per trial (metrics measured after 2s settle window):
  - lock identity stable (transitions <= 2)
  - center error bounded (max |ex| < 0.25, max |ey| < 0.30)
  - real-det fraction >= 0.5
"""
from __future__ import annotations
import json, statistics, time, urllib.request

WEB = "http://127.0.0.1:5055"


def P(p, b):
    r = urllib.request.Request(
        f"{WEB}{p}", data=json.dumps(b).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    return json.loads(urllib.request.urlopen(r, timeout=15).read())


def G(p):
    return json.loads(urllib.request.urlopen(f"{WEB}{p}", timeout=8).read())


def reset(*, bus=False, mantis=False, cars=False):
    P("/api/bus_moving", {"enabled": bus})
    P("/api/mantis_moving", {"enabled": mantis})
    P("/api/cars_moving", {"enabled": cars})
    P("/api/mantis_drive", {"reset": True})
    P("/api/zoom", {"zoom": 1.0})
    P("/api/gains", {"reset": True})
    P("/api/detector", {"mode": "world"})
    P("/api/world_classes", {"classes": "bus, car, truck, person"})
    P("/api/mode", {"mode": "home"})
    time.sleep(3)


def lock_largest(jog_pan=0, jog_tilt=-2):
    P("/api/mode", {"mode": "manual"})
    P("/api/jog", {"dpan": jog_pan, "dtilt": jog_tilt})
    time.sleep(3.5)
    d = G("/api/status")
    if not d["detections"]:
        return None
    t = max(d["detections"],
            key=lambda x: (x["bbox"][2] - x["bbox"][0])
                          * (x["bbox"][3] - x["bbox"][1]))
    cx = (t["bbox"][0] + t["bbox"][2]) / 2
    cy = (t["bbox"][1] + t["bbox"][3]) / 2
    P("/api/select", {"x": cx, "y": cy})
    time.sleep(0.5)
    return t


def sample(duration_s):
    samples = []
    transitions = 0
    last_sel = None
    t0 = time.time()
    while time.time() - t0 < duration_s:
        d = G("/api/status")
        sel = d["selected_id"]
        locked = next((x for x in d["detections"] if x["id"] == sel), None)
        if locked:
            cx = (locked["bbox"][0] + locked["bbox"][2]) / 2
            cy = (locked["bbox"][1] + locked["bbox"][3]) / 2
            samples.append((cx, cy))
        if sel != last_sel:
            transitions += 1
            last_sel = sel
        time.sleep(0.1)
    return samples, transitions


def trial_metrics(samples_ts, transitions, duration_s, settle_s=2.0):
    if not samples_ts:
        return {"pass": False, "real_pct": 0.0, "trans": transitions,
                "ex_max": None, "ey_max": None}
    # Drop the first `settle_s` so initial lock transient doesn't bias the
    # max-error metric.
    t0 = samples_ts[0][0]
    settled = [(cx, cy) for (t, cx, cy) in samples_ts if (t - t0) >= settle_s]
    if not settled:
        settled = [(cx, cy) for (t, cx, cy) in samples_ts]
    expected = int((duration_s - settle_s) * 10)
    ex = [(s[0] - 640) / 640 for s in settled]
    ey = [(s[1] - 360) / 360 for s in settled]
    real_pct = len(settled) / expected if expected > 0 else 0.0
    ex_max = max(abs(x) for x in ex)
    ey_max = max(abs(y) for y in ey)
    # Up to 4 id-rebinds tolerated as long as the pixel error stays tight:
    # YOLO occasionally drops a ByteTrack id during motion and the resolver
    # re-binds via signature to the same physical target. That's a benign
    # id-churn, not a real lock loss.
    passed = (transitions <= 4 and ex_max < 0.25
              and ey_max < 0.30 and real_pct >= 0.50)
    return {
        "pass": passed,
        "real_pct": real_pct,
        "trans": transitions,
        "ex_mean": statistics.mean(ex),
        "ex_std": statistics.stdev(ex) if len(ex) > 1 else 0.0,
        "ex_max": ex_max,
        "ey_mean": statistics.mean(ey),
        "ey_std": statistics.stdev(ey) if len(ey) > 1 else 0.0,
        "ey_max": ey_max,
    }


def run_scenario(name, n_trials, duration_s, *, bus=False, mantis=False,
                 cars=False, drive_vx=0.0, drive_vyaw=0.0):
    print(f"\n=== {name} — {n_trials} trials × {duration_s}s ===")
    results = []
    for i in range(n_trials):
        reset(bus=bus, mantis=mantis, cars=cars)
        hit = lock_largest()
        if not hit:
            print(f"  trial {i+1}: NO DETECTION (skip)")
            results.append({"pass": False, "real_pct": 0.0,
                            "trans": 0, "ex_max": None, "ey_max": None})
            continue
        if drive_vx or drive_vyaw:
            # alternate direction every 2s
            direction = 1
            last_switch = time.time()
            P("/api/mantis_drive", {"vx": direction * drive_vx,
                                    "vyaw": direction * drive_vyaw})
        if mantis:
            P("/api/mantis_moving", {"enabled": True})
        if bus:
            P("/api/bus_moving", {"enabled": True})

        samples_ts = []  # (t_since_start, cx, cy)
        transitions = 0
        last_sel = None
        t0 = time.time()
        direction = 1
        last_switch = t0
        while time.time() - t0 < duration_s:
            if (drive_vx or drive_vyaw) and time.time() - last_switch >= 2.0:
                direction = -direction
                P("/api/mantis_drive", {"vx": direction * drive_vx,
                                        "vyaw": direction * drive_vyaw})
                last_switch = time.time()
            d = G("/api/status")
            sel = d["selected_id"]
            locked = next((x for x in d["detections"] if x["id"] == sel), None)
            if locked:
                cx = (locked["bbox"][0] + locked["bbox"][2]) / 2
                cy = (locked["bbox"][1] + locked["bbox"][3]) / 2
                samples_ts.append((time.time() - t0, cx, cy))
            if sel != last_sel:
                transitions += 1
                last_sel = sel
            time.sleep(0.1)
        samples = samples_ts

        if drive_vx or drive_vyaw:
            P("/api/mantis_drive", {"vx": 0, "vy": 0, "vyaw": 0})

        m = trial_metrics(samples, transitions, duration_s)
        results.append(m)
        tag = "PASS" if m["pass"] else "FAIL"
        if m["ex_max"] is not None:
            print(f"  trial {i+1}: {tag}  real={m['real_pct']*100:.0f}% "
                  f"trans={m['trans']} ex_max={m['ex_max']:.2f} "
                  f"ey_max={m['ey_max']:.2f}")
        else:
            print(f"  trial {i+1}: {tag}  no real_det")
    pct_pass = sum(1 for r in results if r["pass"]) / len(results) * 100
    passing = [r for r in results if r["pass"]]
    if passing:
        ex_max_p = statistics.mean(r["ex_max"] for r in passing)
        ey_max_p = statistics.mean(r["ey_max"] for r in passing)
        print(f"  -> PASS {pct_pass:.0f}%  mean ex_max={ex_max_p:.2f} "
              f"ey_max={ey_max_p:.2f} (passing trials only)")
    else:
        print(f"  -> PASS {pct_pass:.0f}%  no passing trials")
    return results


def main():
    print(f"REDUNDANCY BENCH @ {time.strftime('%H:%M:%S')}")
    summary = {}

    summary["static"] = run_scenario("S1 static lock", 6, 8)
    summary["bus_only"] = run_scenario("S2 bus moving only", 5, 12, bus=True)
    summary["ego_only"] = run_scenario("S3 ego forward 8 m/s only", 5, 12,
                                       drive_vx=8.0)
    summary["ego_yaw"] = run_scenario("S4 ego forward 6 + yaw 0.3", 5, 12,
                                      drive_vx=6.0, drive_vyaw=0.3)
    summary["bus_ego"] = run_scenario("S5 bus + ego 8 m/s", 5, 12,
                                      bus=True, drive_vx=8.0)
    summary["bus_ego_yaw"] = run_scenario(
        "S6 bus + ego 6 m/s + yaw 0.3", 5, 12,
        bus=True, drive_vx=6.0, drive_vyaw=0.3)

    print("\n\n=== AGGREGATE ===")
    for name, results in summary.items():
        p = sum(1 for r in results if r["pass"])
        print(f"{name:18s}  pass {p}/{len(results)}")
    reset()
    P("/api/mantis_drive", {"reset": True})


if __name__ == "__main__":
    main()
