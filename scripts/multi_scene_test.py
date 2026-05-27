#!/usr/bin/env python3
"""Exercise the controller across multiple scenarios and print a report.

Scenarios:
  S1  static-lock                  baseline
  S2  dynamic-cars-only            tracking a moving car
  S3  dynamic-multi                cars + balls all moving
  S4  zoom-stress                  zoom 1->2.5->1 mid-track
  S5  retarget                     lock A, switch to B mid-track
"""
from __future__ import annotations
import json, statistics, time, urllib.request

WEB = "http://127.0.0.1:5055"


def P(p, b):
    r = urllib.request.Request(
        f"{WEB}{p}", data=json.dumps(b).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(r, timeout=10).read())


def G(p):
    return json.loads(urllib.request.urlopen(f"{WEB}{p}", timeout=8).read())


def reset():
    P("/api/zoom", {"zoom": 1.0})
    P("/api/cars_moving", {"enabled": False})
    P("/api/people_walking", {"enabled": False})
    P("/api/moving_target", {"enabled": False})
    P("/api/mode", {"mode": "home"})
    time.sleep(2.5)


def lock_first(view_jog_pan=22, view_jog_tilt=-3, want_name=None):
    P("/api/mode", {"mode": "manual"})
    P("/api/jog", {"dpan": view_jog_pan, "dtilt": view_jog_tilt})
    time.sleep(5)
    d = G("/api/status")
    if not d["detections"]:
        return None
    if want_name:
        cand = [x for x in d["detections"] if x["name"] == want_name]
        t = cand[0] if cand else max(d["detections"], key=lambda x: x["score"])
    else:
        t = max(d["detections"], key=lambda x: x["score"])
    cx = (t["bbox"][0] + t["bbox"][2]) / 2
    cy = (t["bbox"][1] + t["bbox"][3]) / 2
    P("/api/select", {"x": cx, "y": cy})
    return t


def collect(duration_s, hz=4):
    out = []
    n = int(duration_s * hz)
    sel_id_init = None
    for i in range(n):
        d = G("/api/status")
        if sel_id_init is None and d["selected_id"] is not None:
            sel_id_init = d["selected_id"]
        det = next((x for x in d["detections"] if x["id"] == d["selected_id"]), None)
        if det:
            cx = (det["bbox"][0] + det["bbox"][2]) / 2
            cy = (det["bbox"][1] + det["bbox"][3]) / 2
            out.append((i / hz, (cx - 640) / 640, (cy - 360) / 360,
                        d["selected_id"], d["selected_name"]))
        else:
            out.append((i / hz, None, None, d["selected_id"], d["selected_name"]))
        time.sleep(1.0 / hz)
    return out, sel_id_init


def stats(samples, label, sel_init):
    real = [s for s in samples if s[1] is not None]
    if not real:
        return f"  {label}: NO real detections in {len(samples)} samples"
    ids = {s[3] for s in samples if s[3] is not None}
    later = [s for s in real if s[0] > 3]
    if later:
        ex = [s[1] for s in later]; ey = [s[2] for s in later]
        ss = f"ex={statistics.mean(ex):+.3f}±{(statistics.stdev(ex) if len(ex)>1 else 0):.3f} ey={statistics.mean(ey):+.3f}±{(statistics.stdev(ey) if len(ey)>1 else 0):.3f}"
    else:
        ss = "(no late samples)"
    converge = None
    for j in range(len(real) - 2):
        if all((real[k][1] ** 2 + real[k][2] ** 2) ** 0.5 < 0.05 for k in (j, j + 1, j + 2)):
            converge = real[j][0]; break
    return f"  {label}: real={len(real)}/{len(samples)} init_id={sel_init} unique_ids={ids} lock_t={converge} SS {ss}"


def main():
    reports = []

    # S1 static
    reset()
    t = lock_first(22, -3)
    if t:
        samples, sid = collect(8, 4)
        reports.append(stats(samples, "S1 static", sid))
    else:
        reports.append("  S1 static: no detection")

    # S2 dynamic-cars
    reset()
    P("/api/cars_moving", {"enabled": True})
    time.sleep(2)
    t = lock_first(22, -3, want_name="car")
    if t:
        samples, sid = collect(10, 4)
        reports.append(stats(samples, "S2 dynamic-car", sid))
    P("/api/cars_moving", {"enabled": False})

    # S3 dynamic-multi (cars + balls)
    reset()
    P("/api/cars_moving", {"enabled": True})
    P("/api/people_walking", {"enabled": True})
    P("/api/moving_target", {"enabled": True})
    time.sleep(2)
    t = lock_first(22, -3)
    if t:
        samples, sid = collect(10, 4)
        reports.append(stats(samples, "S3 multi", sid))
    P("/api/cars_moving", {"enabled": False})
    P("/api/people_walking", {"enabled": False})
    P("/api/moving_target", {"enabled": False})

    # S4 zoom-stress
    reset()
    P("/api/cars_moving", {"enabled": True})
    time.sleep(2)
    t = lock_first(22, -3, want_name="car")
    if t:
        time.sleep(3)
        P("/api/zoom", {"zoom": 2.5})
        time.sleep(4)
        P("/api/zoom", {"zoom": 1.0})
        samples, sid = collect(5, 4)
        reports.append(stats(samples, "S4 zoom-stress", sid))
    P("/api/zoom", {"zoom": 1.0})
    P("/api/cars_moving", {"enabled": False})

    # S5 retarget
    reset()
    P("/api/cars_moving", {"enabled": True})
    time.sleep(2)
    t = lock_first(22, -3)
    time.sleep(3)
    d = G("/api/status")
    others = [x for x in d["detections"] if x["id"] != d.get("selected_id")]
    if others:
        o = max(others, key=lambda x: x["score"])
        ox = (o["bbox"][0] + o["bbox"][2]) / 2
        oy = (o["bbox"][1] + o["bbox"][3]) / 2
        P("/api/select", {"x": ox, "y": oy})
        samples, sid = collect(6, 4)
        reports.append(stats(samples, "S5 retarget", sid))
    else:
        reports.append("  S5 retarget: only one target in view, skipped")
    P("/api/cars_moving", {"enabled": False})

    print("\n=== MULTI-SCENE TRIAL REPORT ===")
    for r in reports:
        print(r)


if __name__ == "__main__":
    main()
