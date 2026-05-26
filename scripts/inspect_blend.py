#!/usr/bin/env python3
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector


def world_bounds(obj):
    if not hasattr(obj, "bound_box") or not obj.bound_box:
        return None
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    mn = [min(c[i] for c in corners) for i in range(3)]
    mx = [max(c[i] for c in corners) for i in range(3)]
    return [mn, mx]


items = []
for obj in bpy.context.scene.objects:
    constraints = []
    for con in obj.constraints:
        data = {
            "name": con.name,
            "type": con.type,
        }
        for attr in (
            "use_limit_x", "use_limit_y", "use_limit_z",
            "min_x", "max_x", "min_y", "max_y", "min_z", "max_z",
            "owner_space",
        ):
            if hasattr(con, attr):
                value = getattr(con, attr)
                if isinstance(value, float):
                    value = math.degrees(value)
                data[attr] = value
        constraints.append(data)
    items.append({
        "name": obj.name,
        "type": obj.type,
        "parent": obj.parent.name if obj.parent else None,
        "location": [round(v, 6) for v in obj.location],
        "rotation_euler_deg": [round(math.degrees(v), 6) for v in obj.rotation_euler],
        "scale": [round(v, 6) for v in obj.scale],
        "bounds": world_bounds(obj),
        "constraints": constraints,
        "modifiers": [{"name": m.name, "type": m.type} for m in obj.modifiers],
    })

print(json.dumps({
    "file": bpy.data.filepath,
    "frame_start": bpy.context.scene.frame_start,
    "frame_end": bpy.context.scene.frame_end,
    "objects": items,
}, indent=2))
