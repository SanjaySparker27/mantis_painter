# MANTIS PAINTER

Realtime persistent tracking with precision control.

MANTIS PAINTER is a fixed-place Gazebo simulation for educational perception, tracking, and pan/tilt control research.

It uses the provided Blender-derived MANTIS geometry, a real Gazebo nose camera topic, object detection overlays in a browser Web UI, and joint commands for the pan and tilt axes. It does not simulate projectile physics or real-world deployment behavior. The only event output is a non-physical `virtual_mark` when a selected target is held near the camera center.

## Current System

- Simulator: Gazebo Harmonic / gz-sim.
- Robot model: `models/mantis_robot/model.sdf`.
- World: `worlds/mantis_robot_world.sdf`.
- Live camera topic: `/mantis/nose_camera/image`.
- Pan command topic: `/mantis/pan_cmd`.
- Tilt command topic: `/mantis/tilt_cmd`.
- Web UI: `http://127.0.0.1:5055`.
- Source Blender file: `/home/sanju/Documents/BLENDER/MANTIS.blend`.

Extracted rotation limits from the Blender file:

- Pan: `-85.3` to `+89.2` degrees.
- Tilt: `-40.0` to `+30.0` degrees.

## Run

Terminal 1:

```bash
cd /home/sanju/MANTIS_PAINTER
gz sim -v 3 worlds/mantis_robot_world.sdf
```

Terminal 2:

```bash
cd /home/sanju/MANTIS_PAINTER
python3 web_app.py
```

Open:

```text
http://127.0.0.1:5055
```

Use the browser UI to:

- View the real Gazebo nose-camera feed.
- See colored target detections and bounding boxes.
- Click a detected object or select it from the detections table.
- Let the controller center the selected object using pan and tilt.
- Use `Home / clear target` to cancel selection and return to the forward home pose.

## Important Files

- `web_app.py`: Flask Web UI, Gazebo image subscriber, color detector, and bounded PID-style pan/tilt controller.
- `worlds/mantis_robot_world.sdf`: main world with road, vehicles, target markers, MANTIS robot, camera, and GUI plugins.
- `models/mantis_robot/model.sdf`: robot links, joints, limits, nose camera, and Gazebo joint position controllers.
- `scripts/export_mantis_robot.py`: exports Blender objects into simulation meshes and SDF.
- `scripts/inspect_blend.py`: inspects Blender object hierarchy and rotation constraints.
- `docs/MECHANISM_AND_3D_ASSETS.md`: exact 3D files, Gazebo links, joints, limits, topics, and how rotation is controlled.
- `docs/RESEARCH_UPGRADE_PLAN.md`: suggested non-harmful research upgrades.
- `docs/SAFE_CLAUDE_CODE_PROMPT.md`: prompt template for using Claude Code safely.

## Controller Notes

The controller is a real-`dt` PID with FOV-aware error mapping:

- Image error is converted to a true off-axis angle via `atan(tan(FOV/2)*nx)`.
- PID runs at the real camera frame rate (no fixed `CONTROL_HZ` assumption).
- Anti-windup clamp on the integral, derivative on smoothed pixel error.
- Output is a degree correction; the actual joint step is rate-limited.
- Target identity is held across frames by name + nearest-anchor matching so
  multiple same-color targets do not cause the track to jump.
- Modes: `auto` (track), `manual` (jog only), `home` (smooth return to 0, 12).
- Lost target: holds for 0.8 s, then auto-clears and homes.

Web UI controls:

- Mode buttons, Clear target.
- Jog pad with selectable step (0.5°–10°).
- Keyboard: arrows / WASD jog, Space = home, C = clear, M = manual, T = auto.
- Live sliders for `Kp`, `Ki`, `Kd`, `max_rate`, `deadband`.

Joint position controllers in `model.sdf` were retuned (pan p=900 d=500, tilt
p=700 d=360) because the previous gains (p=8, d=0.25) left the joints
underdamped and let pan drift even with a zero command.

Tuning workflow:

- Reduce `deadband` for tighter centering, raise it if jitter appears.
- Raise `Kp` until response is fast but not oscillating.
- Add `Kd` until overshoot disappears.
- Keep `Ki` small; it is only for steady offset.
- Keep `max_rate` realistic (deg/s) so the joint can follow.

## Safe Scope

Keep this project focused on:

- Robotic perception.
- Camera simulation.
- Multi-object tracking.
- Pan/tilt servo control.
- Evaluation metrics.
- Non-physical virtual marking.

Avoid adding:

- Real firing, impact, or projectile models.
- Instructions for constructing or deploying a physical launcher.
- Autonomous engagement logic outside a closed educational simulation.
