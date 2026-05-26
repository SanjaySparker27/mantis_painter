# Research Upgrade Plan

This plan keeps MANTIS PAINTER in a safe educational scope: perception, tracking, pan/tilt control, simulation fidelity, and metrics. Treat all target events as virtual annotations.

## 1. Make The Camera And Robot Geometry More Accurate

- Re-export the Blender model whenever the pan/tilt hierarchy changes.
- Keep one SDF link for the fixed base, one for the yaw body, and one for the tilting nose.
- Place the camera frame exactly where the physical sensor would be mounted.
- Validate camera orientation by spawning a checkerboard or colored target at known world positions.
- Record expected pixel movement for positive pan and positive tilt so controller signs are documented.

## 2. Improve The Scene

- Add several static vehicles at different distances and angles.
- Add occluders such as poles, signs, parked vehicles, and road barriers.
- Add lighting presets: daylight, overcast, dusk, and high-glare.
- Add target colors or AprilTag-style calibration boards for debugging.
- Keep the main MANTIS platform fixed in place for repeatable tests.

## 3. Replace Color Detection With A Real Detector

Recommended progression:

1. Keep HSV color detection as the debug baseline.
2. Add OpenCV DNN or ONNX Runtime with a vehicle detector.
3. Add YOLO only for classifying vehicles in the simulated camera feed.
4. Compare detector latency, false positives, and missed detections.

Keep the detector output limited to bounding boxes, class names, confidence, and track IDs.

## 4. Add Multi-Object Tracking

Start simple:

- Use IoU matching between consecutive frames.
- Add a constant-velocity Kalman filter.
- Add track age, lost-frame count, and confidence smoothing.

Then evaluate stronger options:

- ByteTrack for detector association.
- BoT-SORT if appearance embeddings are needed.

Metrics to log:

- Track continuity.
- ID switches.
- Time to first lock.
- Centering error in pixels.
- Pan and tilt command smoothness.

## 5. Tune Pan/Tilt Control

Use the current controller in `web_app.py` as the baseline.

Suggested workflow:

1. Disable integral gain.
2. Tune proportional gain until the target moves toward center quickly.
3. Add derivative damping until overshoot is acceptable.
4. Add only a small integral term for steady offset.
5. Keep rate limits realistic.
6. Log every run before changing another parameter.

Useful plots:

- Target pixel error X/Y over time.
- Pan/tilt angle over time.
- Pan/tilt rate over time.
- Selected target confidence over time.
- Lost/reacquired target timestamps.

## 6. Improve The Web UI

- Add a graph panel for pixel error, pan angle, and tilt angle.
- Add target thumbnails from recent frames.
- Add a target list with stable track IDs.
- Add manual jog controls for pan/tilt testing.
- Add a controller mode selector: manual, auto-center, home.
- Add run export as CSV and JSON.

## 7. ROS 2 Integration

When ready, split the system into ROS 2 nodes:

- Camera bridge node.
- Detector node.
- Tracker node.
- Controller node.
- Metrics logger node.
- Web dashboard bridge.

Use ROS messages for bounding boxes, tracks, selected target, and joint commands.

## 8. Higher Fidelity Simulation

Only move to heavier simulators after the NVIDIA driver works correctly.

Options:

- Gazebo with better models and lighting for robotics workflow.
- CARLA for traffic-heavy vehicle scenes.
- Isaac Sim only if GPU drivers and storage are stable.

Keep Gazebo as the baseline because it is already running on this machine.
