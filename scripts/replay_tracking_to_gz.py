#!/usr/bin/env python3
"""Replay pan/tilt tracking commands into Gazebo joint-position controllers."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def publish(topic: str, value: float) -> None:
    subprocess.run(
        ["gz", "topic", "-t", topic, "-m", "gz.msgs.Double", "-p", f"data: {value:.8f}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "output" / "tracking_metrics.csv")
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.csv.open()))
    step = max(1, round(30 / args.rate))
    delay = 1.0 / args.rate

    while True:
        for row in rows[::step]:
            pan = math.radians(float(row["pan_deg"]))
            tilt = math.radians(float(row["tilt_deg"]))
            publish("/mantis/pan_cmd", pan)
            publish("/mantis/tilt_cmd", tilt)
            time.sleep(delay)
        if not args.loop:
            break


if __name__ == "__main__":
    main()
