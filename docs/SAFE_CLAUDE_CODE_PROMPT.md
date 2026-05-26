# Safe Claude Code Prompt

Use this prompt when asking Claude Code to improve the project. It keeps the work focused on educational simulation, perception, tracking, pan/tilt robotics, and non-physical virtual marking.

```text
You are working in /home/sanju/MANTIS_PAINTER.

This is an educational Gazebo robotics simulation called MANTIS PAINTER. Keep the project strictly focused on safe perception and control research:

- simulated camera feeds
- object detection bounding boxes
- multi-object tracking
- pan/tilt joint control
- Web UI visualization
- logging and evaluation metrics
- non-physical virtual mark events only

Do not add projectile physics, firing logic, real-world launcher construction instructions, impact modeling, or autonomous engagement behavior. Treat every target interaction as a virtual annotation inside the simulator.

Current project structure:

- web_app.py: Flask Web UI, Gazebo camera subscriber, detector, target selection, pan/tilt controller.
- worlds/mantis_robot_world.sdf: main Gazebo world.
- models/mantis_robot/model.sdf: MANTIS robot with pan/tilt joints and nose camera.
- scripts/export_mantis_robot.py: Blender-to-Gazebo export script.
- docs/RESEARCH_UPGRADE_PLAN.md: safe upgrade plan.

Task:

1. Read the relevant files first.
2. Preserve the fixed static MANTIS platform.
3. Improve only the safe simulation feature I request.
4. Keep changes small and testable.
5. Run syntax checks and any available local verification.
6. Summarize changed files and exact run commands.

Feature request:

[PUT ONE SAFE FEATURE HERE, for example:
- Add a graph of target center error, pan angle, and tilt angle to the Web UI.
- Add stable track IDs using IoU matching and lost-frame handling.
- Add manual pan/tilt jog buttons to the UI.
- Add CSV logging for detections, selected track, pan, tilt, and pixel error.
- Add a calibration target world and document controller sign checks.]
```

## Example Safe Requests

```text
Add stable object tracking IDs to web_app.py using IoU matching. Keep outputs limited to bbox, class name, confidence, and track ID. Do not add any projectile or real-world actuation behavior.
```

```text
Add a Web UI graph showing pixel error X/Y, pan angle, and tilt angle over the last 20 seconds. Use the existing /api/status data or add a safe metrics endpoint.
```

```text
Create a Gazebo calibration world with colored boxes at known positions so I can verify camera orientation and pan/tilt controller signs.
```

```text
Add CSV logging for educational evaluation: timestamp, selected target, bbox center, pixel error, pan_deg, tilt_deg, and whether a virtual mark event occurred.
```

## Avoid These Requests

Do not ask for:

- launcher construction
- firing mechanisms
- projectile trajectories
- hit/impact optimization
- real-world autonomous targeting
- bypassing safety restrictions

Reframe those as:

- virtual annotation events
- camera centering evaluation
- robotics servo control
- detector and tracker benchmarking
