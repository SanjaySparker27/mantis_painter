#!/usr/bin/env python3
"""Convert the single MANTIS STL into simulation-oriented visual parts.

The source STL does not contain link/object names. This script keeps the real
mesh triangles but partitions them by spatial heuristics into:

- base: lower fixed platform / lower details
- pan_body: upper body that follows yaw
- tilt_nose: forward nose section that follows pitch
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models" / "mantis" / "meshes" / "MANTIS.stl"
OUT = ROOT / "models" / "mantis" / "converted"


def export_part(mesh: trimesh.Trimesh, mask: np.ndarray, name: str) -> dict:
    part = mesh.submesh([np.flatnonzero(mask)], append=True, repair=False)
    path = OUT / f"{name}.stl"
    part.export(path)
    return {
        "file": f"/assets/mantis/{name}.stl",
        "faces": int(len(part.faces)),
        "bounds": part.bounds.tolist() if len(part.faces) else None,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(SRC, force="mesh")
    centers = mesh.triangles_center
    bounds = mesh.bounds
    model_center = bounds.mean(axis=0)

    # The screenshots show the long front/nose section extending toward -X.
    # Keep a conservative split so the front section can pitch without moving
    # the whole upper hull.
    z = centers[:, 2]
    x = centers[:, 0]
    upper = z > 1.18
    tilt_nose = upper & (x < -1.15)
    pan_body = upper & ~tilt_nose
    base = ~upper

    # Make sure every triangle is assigned exactly once.
    assert np.all(base | pan_body | tilt_nose)
    assert not np.any(base & pan_body)
    assert not np.any(base & tilt_nose)
    assert not np.any(pan_body & tilt_nose)

    metadata = {
        "source": str(SRC),
        "model_center": model_center.tolist(),
        "pan_pivot": [float(model_center[0]), float(model_center[1]), 1.22],
        "tilt_pivot": [-1.15, float(model_center[1]), 1.45],
        "parts": {
            "base": export_part(mesh, base, "base"),
            "pan_body": export_part(mesh, pan_body, "pan_body"),
            "tilt_nose": export_part(mesh, tilt_nose, "tilt_nose"),
        },
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
