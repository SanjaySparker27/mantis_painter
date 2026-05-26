#!/usr/bin/env python3
"""Export MANTIS Blender parts as link-local meshes and an SDF robot model."""

from __future__ import annotations

import json
import math
from pathlib import Path

import bpy
import bmesh
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "mantis_robot"
MESH_DIR = MODEL_DIR / "meshes"
MESH_DIR.mkdir(parents=True, exist_ok=True)

BASE = "BAREL_standardSurface1_0"
PAN = "BAREL_standardSurface1_0.001"
TILT = "BAREL_standardSurface1_0.002"


def vec(obj_name: str) -> Vector:
    return bpy.data.objects[obj_name].matrix_world.translation.copy()


def limit(obj_name: str, axis: str) -> tuple[float, float]:
    obj = bpy.data.objects[obj_name]
    for con in obj.constraints:
        if con.type == "LIMIT_ROTATION":
            return getattr(con, f"min_{axis}"), getattr(con, f"max_{axis}")
    raise RuntimeError(f"No LIMIT_ROTATION on {obj_name}")


def write_ascii_stl(obj_name: str, origin: Vector, out_path: Path) -> dict:
    obj = bpy.data.objects[obj_name]
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.normal_update()

    verts = []
    with out_path.open("w") as f:
        f.write(f"solid {obj_name}\n")
        for face in bm.faces:
            normal = face.normal
            f.write(f"  facet normal {normal.x:.8e} {normal.y:.8e} {normal.z:.8e}\n")
            f.write("    outer loop\n")
            for vert in face.verts:
                world = obj.matrix_world @ vert.co
                local = world - origin
                verts.append(local.copy())
                f.write(f"      vertex {local.x:.8e} {local.y:.8e} {local.z:.8e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {obj_name}\n")

    bm.free()
    eval_obj.to_mesh_clear()

    mins = [min(v[i] for v in verts) for i in range(3)]
    maxs = [max(v[i] for v in verts) for i in range(3)]
    return {"file": f"meshes/{out_path.name}", "bounds": [mins, maxs], "triangles": len(verts) // 3}


def sdf_pose(v: Vector) -> str:
    return f"{v.x:.6f} {v.y:.6f} {v.z:.6f} 0 0 0"


def main() -> None:
    base_origin = Vector((0, 0, 0))
    pan_origin = vec(PAN)
    tilt_origin = vec(TILT)

    pan_min, pan_max = limit(PAN, "z")
    tilt_min, tilt_max = limit(TILT, "y")

    meshes = {
        "base": write_ascii_stl(BASE, base_origin, MESH_DIR / "base_link.stl"),
        "pan": write_ascii_stl(PAN, pan_origin, MESH_DIR / "pan_link.stl"),
        "tilt": write_ascii_stl(TILT, tilt_origin, MESH_DIR / "tilt_link.stl"),
    }

    metadata = {
        "source": bpy.data.filepath,
        "objects": {"base": BASE, "pan": PAN, "tilt": TILT},
        "origins": {
            "base": list(base_origin),
            "pan": list(pan_origin),
            "tilt": list(tilt_origin),
        },
        "limits_rad": {
            "pan": [pan_min, pan_max],
            "tilt": [tilt_min, tilt_max],
        },
        "limits_deg": {
            "pan": [math.degrees(pan_min), math.degrees(pan_max)],
            "tilt": [math.degrees(tilt_min), math.degrees(tilt_max)],
        },
        "meshes": meshes,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    tilt_camera_pose = "6.250000 0.000000 1.650000 0 0 0"
    rel_tilt = tilt_origin - pan_origin
    model_sdf = f"""<?xml version="1.0" ?>
<sdf version="1.10">
  <model name="mantis_robot">
    <link name="base_link">
      <pose>{sdf_pose(base_origin)}</pose>
      <inertial><mass>80</mass><inertia><ixx>45</ixx><iyy>45</iyy><izz>45</izz></inertia></inertial>
      <collision name="base_collision">
        <pose>0.9 -2.37 0.65 0 0 0</pose>
        <geometry><box><size>9.4 9.4 1.4</size></box></geometry>
      </collision>
      <visual name="base_visual">
        <geometry><mesh><uri>meshes/base_link.stl</uri></mesh></geometry>
        <material><ambient>0.18 0.20 0.22 1</ambient><diffuse>0.26 0.29 0.32 1</diffuse></material>
      </visual>
    </link>

    <link name="pan_link">
      <pose>{sdf_pose(pan_origin)}</pose>
      <inertial><mass>40</mass><inertia><ixx>25</ixx><iyy>25</iyy><izz>25</izz></inertia></inertial>
      <collision name="pan_collision">
        <pose>0 0 1.1 0 0 0</pose>
        <geometry><box><size>8.2 6.2 2.8</size></box></geometry>
      </collision>
      <visual name="pan_visual">
        <geometry><mesh><uri>meshes/pan_link.stl</uri></mesh></geometry>
        <material><ambient>0.08 0.10 0.12 1</ambient><diffuse>0.12 0.15 0.18 1</diffuse></material>
      </visual>
    </link>

    <joint name="pan_joint" type="revolute">
      <parent>base_link</parent>
      <child>pan_link</child>
      <pose relative_to="base_link">{sdf_pose(pan_origin)}</pose>
      <axis>
        <xyz expressed_in="base_link">0 0 1</xyz>
        <limit><lower>{pan_min:.8f}</lower><upper>{pan_max:.8f}</upper><effort>500</effort><velocity>1.2</velocity></limit>
        <dynamics><damping>0.12</damping><friction>0.03</friction></dynamics>
      </axis>
    </joint>

    <link name="tilt_link">
      <pose>{sdf_pose(tilt_origin)}</pose>
      <inertial><mass>12</mass><inertia><ixx>8</ixx><iyy>8</iyy><izz>8</izz></inertia></inertial>
      <collision name="tilt_collision">
        <pose>3.0 0 0.6 0 0 0</pose>
        <geometry><box><size>6.4 1.8 1.8</size></box></geometry>
      </collision>
      <visual name="tilt_visual">
        <geometry><mesh><uri>meshes/tilt_link.stl</uri></mesh></geometry>
        <material><ambient>0.45 0.48 0.52 1</ambient><diffuse>0.60 0.63 0.68 1</diffuse></material>
      </visual>
      <visual name="nose_camera_marker">
        <pose>{tilt_camera_pose}</pose>
        <geometry><box><size>0.28 0.22 0.18</size></box></geometry>
        <material><ambient>0.0 0.55 0.65 1</ambient><diffuse>0.0 0.8 0.9 1</diffuse></material>
      </visual>
      <sensor name="nose_camera" type="camera">
        <pose>{tilt_camera_pose}</pose>
        <topic>/mantis/nose_camera/image</topic>
        <update_rate>30</update_rate>
        <camera>
          <horizontal_fov>1.012</horizontal_fov>
          <image><width>1280</width><height>720</height><format>R8G8B8</format></image>
          <clip><near>0.1</near><far>160</far></clip>
        </camera>
      </sensor>
    </link>

    <joint name="tilt_joint" type="revolute">
      <parent>pan_link</parent>
      <child>tilt_link</child>
      <pose relative_to="pan_link">{sdf_pose(rel_tilt)}</pose>
      <axis>
        <xyz expressed_in="pan_link">0 1 0</xyz>
        <limit><lower>{tilt_min:.8f}</lower><upper>{tilt_max:.8f}</upper><effort>220</effort><velocity>1.0</velocity></limit>
        <dynamics><damping>0.10</damping><friction>0.025</friction></dynamics>
      </axis>
    </joint>

    <plugin filename="gz-sim-joint-position-controller-system" name="gz::sim::systems::JointPositionController">
      <joint_name>pan_joint</joint_name>
      <topic>/mantis/pan_cmd</topic>
      <p_gain>8.0</p_gain>
      <d_gain>0.25</d_gain>
    </plugin>
    <plugin filename="gz-sim-joint-position-controller-system" name="gz::sim::systems::JointPositionController">
      <joint_name>tilt_joint</joint_name>
      <topic>/mantis/tilt_cmd</topic>
      <p_gain>7.0</p_gain>
      <d_gain>0.20</d_gain>
    </plugin>
  </model>
</sdf>
"""
    (MODEL_DIR / "model.sdf").write_text(model_sdf)
    (MODEL_DIR / "model.config").write_text("""<?xml version="1.0"?>
<model>
  <name>MANTIS robotic pan tilt</name>
  <version>1.0</version>
  <sdf version="1.10">model.sdf</sdf>
  <author><name>sanju</name></author>
  <description>MANTIS Blender-derived robotic pan/tilt model with real joints and camera sensor.</description>
</model>
""")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
