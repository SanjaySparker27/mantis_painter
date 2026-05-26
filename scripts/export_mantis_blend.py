#!/usr/bin/env python3
"""Export the MANTIS Blender rig for the Web UI simulation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import bpy


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "mantis" / "blend_export"
OUT.mkdir(parents=True, exist_ok=True)

PAN_OBJECT = "BAREL_standardSurface1_0.001"
TILT_OBJECT = "BAREL_standardSurface1_0.002"


def limit_rotation(obj_name: str) -> dict | None:
    obj = bpy.data.objects[obj_name]
    for con in obj.constraints:
        if con.type != "LIMIT_ROTATION":
            continue
        return {
            "name": con.name,
            "owner_space": con.owner_space,
            "use_limit_x": con.use_limit_x,
            "use_limit_y": con.use_limit_y,
            "use_limit_z": con.use_limit_z,
            "min_x_deg": math.degrees(con.min_x),
            "max_x_deg": math.degrees(con.max_x),
            "min_y_deg": math.degrees(con.min_y),
            "max_y_deg": math.degrees(con.max_y),
            "min_z_deg": math.degrees(con.min_z),
            "max_z_deg": math.degrees(con.max_z),
        }
    return None


metadata = {
    "source": bpy.data.filepath,
    "asset": "/assets/mantis/mantis_rig.glb",
    "objects": {
        "base": "BAREL_standardSurface1_0",
        "pan": PAN_OBJECT,
        "tilt": TILT_OBJECT,
    },
    "limits": {
        "pan": limit_rotation(PAN_OBJECT),
        "tilt": limit_rotation(TILT_OBJECT),
    },
}

for obj in bpy.context.scene.objects:
    obj.select_set(obj.type in {"MESH", "EMPTY"})

bpy.ops.export_scene.gltf(
    filepath=str(OUT / "mantis_rig.glb"),
    export_format="GLB",
    use_selection=True,
    export_apply=False,
    export_yup=True,
    export_animations=False,
)

(OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata, indent=2))
