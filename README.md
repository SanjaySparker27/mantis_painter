# MANTIS PAINTER

Realtime persistent tracking with precision pan/tilt control. Gazebo Harmonic simulation built on top of a Blender-derived MANTIS chassis. Educational scope only — no projectile physics, no real-world deployment instructions. Paint events are virtual signals (gz topic + file) so an external Raspberry Pi or MCU can react over PWM / GPIO if wired.

![nose camera view with object detections, track IDs, crosshair and live PID values](docs/assets/nose_camera.jpg)

### Dynamic tracking — pursuing moving cars

![Live nose-camera view with YOLOv12+ByteTrack: prius locked under the crosshair while a pickup, yellow ball, person and drone are also detected in-frame](docs/assets/nose_camera_dynamic.jpg)

Crosshair is locked at the bbox center of the selected `car` (ID 5).
ByteTrack keeps the ID stable while the prius and pickup drive around;
the controller follows. Other detections (yellow ball ID 50, pickup
ID 18, person, drone) are drawn but inactive.

### Convergence trace — selecting a car, controller centers it

![pan and tilt cmd vs actual on top, normalized pixel error ex/ey on bottom](docs/assets/convergence.png)

Top: blue = pan (cmd solid, actual dashed). Red = tilt (same). Bottom: normalized pixel error of the selected target's bbox center (`ex` horizontal, `ey` vertical). The dotted lines mark the deadband.

## What kind of tracking is this?

Object-level tracking, not blind pixel tracking.

| layer | what it does | implementation |
|---|---|---|
| Detection | finds what objects are in the frame | **OpenCV HSV color masks** (red, blue, green, yellow, cyan, magenta, orange, brown, teal, purple) **or YOLOv12n** (`ultralytics`) on the same frame |
| Track association | keeps a stable identity for the same object across frames | **ByteTrack** (Kalman filter + IoU association) for YOLO mode; **name + nearest-bbox-anchor** with a 240 px gate for color mode |
| Pose control | converts bbox-center pixel error to a joint-angle command | FOV-aware mapping `θ = atan(tan(FOV/2) · n)`, cascaded outer PID on the **actual** joint state, inner Gazebo `JointPositionController` |

So the controller does not chase a pixel — it chases a *tracked object*. If the bbox jitters by 5 px the smoothed center barely moves; if YOLO swaps the IDs of two cars, ByteTrack keeps the one you clicked.

## Features

### Perception
- Live `gz.msgs.Image` subscription on `/mantis/nose_camera/image` (1280×720, 30 Hz, HFOV 1.012 rad)
- HSV color detector with 10+ named color classes
- **YOLOv12n** detector via `ultralytics` (auto-downloaded on first launch)
- **ByteTrack** multi-object tracker with persistent IDs
- Async detection thread — heavy inference never blocks the Flask UI
- Detection score filter (`MIN_TRACK_SCORE = 0.15`) to ignore weak hits

### Tracking
- Real-`dt` PID (not fixed `CONTROL_HZ`)
- FOV-aware pixel-to-angle conversion using `atan(tan(FOV/2) · n)`
- Feedback on **actual** joint position (from `/mantis/joint_states`), not the commanded angle — kills steady-state offset from gravity
- Target-velocity feed-forward with EMA smoothing on bbox centre
- Anti-windup integral clamp, deadband freeze, output low-pass filter
- Lost-target grace (0.8 s) then auto-clear and home
- Identity-stable selection: ByteTrack ID for YOLO, name + nearest-anchor for color

### Control modes
| button | behaviour |
|---|---|
| `Tracking: ON/OFF` | toggle auto-tracking of the selected target |
| `Auto Paint: ON/OFF` | fire one paint pulse whenever the selected target is centred and held. Stays on the same target |
| `Auto Serial Tracker: ON/OFF` | fully autonomous loop: pick next un-painted target → centre → paint → advance. Remembers painted targets across sessions (`/tmp/mantis_painted_memory.json`). Independent of Tracking switch |
| `Reset memory` | clear the painted-target memory |
| `Manual / Jog pad / Arrow keys` | drive pan & tilt directly with step size 0.5°–10° |
| `Home` | smooth return to `pan=0, tilt=12°` |
| `STOP` | freeze cmd at current actual angles |
| `Click-to-Aim` | clicks on the feed aim the camera at that pixel instead of selecting a bbox |
| `PAINT` | one paint pulse on current target. Key `P` |
| `Auto-tune` | step-response FOPDT identification + Cohen-Coon → applies gains |
| `zoom` slider | browser-side digital zoom 1×–4× on the live feed (clicks are corrected back to source coords) |

### Web UI
- Live MJPEG feed at `http://127.0.0.1:5055`
- Overlay: bounding boxes with track IDs + names, crosshair, HUD with pan/tilt/gains/paint count, paint splash animation on trigger
- Live PID sliders that POST to `/api/gains` while you drag (Speed / Hold / Smooth / Max slew / Lock zone)
- Detector toggle YOLO ↔ Color
- Detections table with click-to-select buttons
- Virtual marks history panel
- Status badge with current mode and sweep indicator

### Hardware-out
- Paint events publish `gz.msgs.Int32(pulse_ms)` on `/mantis/paint_trigger`
- Same event appended to `/tmp/mantis_paint.signal` (one line per pulse)
- A Pi or MCU can subscribe to the topic or tail the file and drive a real GPIO/PWM pin. The sim itself does **not** instantiate any projectile or physical actuator.

## Architecture

```mermaid
flowchart LR
  GZ["Gazebo Harmonic<br/>world + sensors"]
  GZ -- "nose_camera/image" --> CAM
  GZ -- "joint_states" --> JS

  subgraph WEBAPP["web_app.py"]
    CAM["image_to_bgr"] --> DETW["detection_worker thread"]
    DETW -- "HSV or YOLO+ByteTrack" --> DETS[("detections")]
    CAM --> CT["control_tick"]
    DETS --> AUTO["auto_control_step"]
    JS --> AUTO
    CT --> AUTO
    CT --> MAN["manual_control_step"]
    CT --> HOME["home_control_step"]
    AUTO --> PUB["publish pan/tilt cmd"]
    AUTO --> PAINTNODE["trigger_paint"]
  end

  PUB -- "gz.msgs.Double" --> GZ
  PAINTNODE -- "gz.msgs.Int32" --> EXT["Pi / MCU GPIO + PWM"]
  PAINTNODE -- "file write" --> SIG["mantis_paint.signal"]
  WEBAPP -- "Flask HTTP" --> BROWSER["Browser UI"]
  BROWSER -- "REST API" --> WEBAPP
```

## Control loop

```mermaid
flowchart TB
  A["bbox center cx, cy"] --> B["EMA smooth (beta = 0.20)"]
  B --> C["normalize to nx, ny in [-1, 1]"]
  C --> D{"inside deadband?"}
  D -- "yes" --> E["freeze cmd toward actual, decay integral"]
  D -- "no" --> F["theta = atan(tan(FOV/2) * n)"]
  F --> G["PID = Kp*theta + Ki*sum + Kd*derivative"]
  JS2["actual joint angle"] --> H
  G --> H["desired = actual + sign * u"]
  H --> I["step = clamp(desired - cmd, +/- max_rate*dt)"]
  I --> J["cmd += lpf * step"]
  J --> K["publish radians on pan_cmd"]
```

## Run

Terminal 1:
```bash
cd /home/sanju/MANTIS_PAINTER
gz sim -v 3 worlds/mantis_robot_world.sdf
```

Terminal 2:
```bash
cd /home/sanju/MANTIS_PAINTER
/home/sanju/venv-ardupilot/bin/python3 web_app.py     # needs flask + gz python + ultralytics
```

Open `http://127.0.0.1:5055`.

## Important files

- `web_app.py` — Flask UI, gz transport subscriber, color+YOLO+ByteTrack detector, cascaded PID with actual-joint feedback, paint trigger
- `worlds/mantis_robot_world.sdf` — road, colored boxes, helipad, ArUco tag, x500 quad, rc_cessna, r1_rover, pickup, prius, standing person, MANTIS robot
- `models/mantis_robot/model.sdf` — pan/tilt joints with analytic-PID JointPositionController, JointStatePublisher, world-fixed base
- `scripts/pid_autotune.py` — standalone CLI Cohen-Coon autotune (the in-app `Auto-tune` button uses the same algorithm in-process)
- `scripts/export_mantis_robot.py` — exports Blender objects into Gazebo meshes + SDF
- `scripts/inspect_blend.py` — inspects Blender hierarchy and rotation constraints
- `docs/MECHANISM_AND_3D_ASSETS.md` — exact 3D file paths, joint limits, topics
- `docs/RESEARCH_UPGRADE_PLAN.md` — suggested non-harmful research upgrades

## Topics

| topic | type | direction |
|---|---|---|
| `/mantis/nose_camera/image` | `gz.msgs.Image` | sim → web_app |
| `/mantis/joint_states` | `gz.msgs.Model` | sim → web_app |
| `/mantis/pan_cmd` | `gz.msgs.Double` (rad) | web_app → sim |
| `/mantis/tilt_cmd` | `gz.msgs.Double` (rad) | web_app → sim |
| `/mantis/paint_trigger` | `gz.msgs.Int32` (pulse ms) | web_app → external |

Joint limits (from Blender source):
- Pan: −85.3° to +89.2°
- Tilt: −40.0° to +30.0°

## Keyboard

| key | action |
|---|---|
| arrows / WASD | jog pan / tilt by selected step |
| Space | Home |
| T | toggle Tracking ON/OFF |
| C | Clear target |
| P | one paint pulse |
| Esc / X | STOP |

## Verified tuning (current default gains)

The defaults committed to `web_app.py` were swept and chosen by
`scripts/auto_tune_trial.py` over multiple gain sets and scenes.

| parameter | value | what it does |
|---|---|---|
| Kp (`Speed` slider) | 0.50 | aggressiveness — how hard to chase pixel error |
| Ki (`Hold` slider) | 0.18 | removes steady-state offset |
| Kd (`Smooth` slider) | 0.14 | damping; clamped derivative ±60°/s |
| max_rate (`Max slew`) | 35 deg/s pan, 26 deg/s tilt; scaled by 1/zoom^1.4 at high zoom |
| deadband (`Lock zone`) | 0.008 (pan), 0.012 (tilt) — controller freezes inside this |
| PID output clamp | 6° per cycle — single-frame outlier can't whiplash the joint |
| LPF on cmd | 0.32, softened by 1/√zoom at high zoom |

Measured performance:

| metric | zoom 1.0× | zoom 2.0× |
|---|---|---|
| Lock time | 1.8 s | 2.5 s |
| SS ex | +0.005 ± 0.004 | −0.018 ± 0.001 |
| SS ey | +0.006 ± 0.006 | −0.019 ± 0.008 |
| Overshoot | none | none |

To re-tune for a different rig: run

```bash
python3 scripts/auto_tune_trial.py
```

It sweeps 5 candidate gain sets across 3 scenes, scores each on
time-to-lock + steady-state error + divergence, and applies the winner.

## Real-world wiring — drive a stepper, servo, or PWM solenoid

The simulation is wired so the same controller can run on a real
turret. A Raspberry Pi or MCU subscribes to one of the output channels
in `/api/channels` and converts each paint pulse + joint angle into
actuator commands.

### Output channels

| channel | enabled by | format | wire it to |
|---|---|---|---|
| `gz_topic` | UI checkbox (default ON) | `gz.msgs.Int32` (pulse_ms) on `/mantis/paint_trigger` | any gz-aware node, ROS2 bridge |
| `file` | UI checkbox (default ON) | line: `time count pulse_ms pan tilt name` appended to `/tmp/mantis_paint.signal` | `tail -F` from a Pi daemon |
| `udp` | UI checkbox | same line as a UDP datagram to `<host>:<port>` | ESP32, MCU on Wi-Fi |
| `tcp` | UI checkbox | one-shot connect + send | server-style actuator daemon |
| `serial` | UI checkbox | pyserial write of the same line | Arduino / Pi GPIO over UART |

The pan and tilt joint targets are also published as `gz.msgs.Double`
on `/mantis/pan_cmd` and `/mantis/tilt_cmd` (radians). Forward those to
your motor driver.

### Example: Raspberry Pi + stepper + servo paintball trigger

```text
+----------------+        +----------------+
| MANTIS PAINTER |--TCP-->| paint_daemon   |    +-------------+
| web_app.py     |--gz-->|  (on the Pi)    |--->| stepper drv | PAN axis (NEMA 17)
|                |        | + RPi.GPIO     |    +-------------+
|                |--gz-->|  + pigpio       |--->| servo PWM   | TILT axis (MG996R)
|                |        |                |    +-------------+
|                |        |                |--->| solenoid    | PAINT (5V relay)
+----------------+        +----------------+    +-------------+
```

Minimal Pi daemon (Python, `pip install pyserial gz-transport13 RPi.GPIO`):

```python
import math, time
import gz.transport13 as gzt
from gz.msgs10.double_pb2 import Double
from gz.msgs10.int32_pb2 import Int32

# Stepper: 200 steps/rev, 1/8 microstepping = 1600 steps / 360 deg
PAN_STEPS_PER_DEG  = 1600 / 360
PAN_DIR_PIN, PAN_STEP_PIN = 23, 24

import RPi.GPIO as GPIO  # not exercised by the sim; wire on the real rig
GPIO.setmode(GPIO.BCM)
GPIO.setup([PAN_DIR_PIN, PAN_STEP_PIN], GPIO.OUT)

import pigpio
pi = pigpio.pi()
TILT_PIN  = 18
PAINT_PIN = 17
pi.set_servo_pulsewidth(TILT_PIN, 1500)

n = gzt.Node()
pan_pos_deg = 0.0

def on_pan(msg: Double):
    global pan_pos_deg
    target = math.degrees(msg.data)
    delta_steps = int((target - pan_pos_deg) * PAN_STEPS_PER_DEG)
    GPIO.output(PAN_DIR_PIN, GPIO.HIGH if delta_steps > 0 else GPIO.LOW)
    for _ in range(abs(delta_steps)):
        GPIO.output(PAN_STEP_PIN, GPIO.HIGH); time.sleep(2e-4)
        GPIO.output(PAN_STEP_PIN, GPIO.LOW);  time.sleep(2e-4)
    pan_pos_deg = target

def on_tilt(msg: Double):
    deg = math.degrees(msg.data)
    # servo pulse: 1.0 ms (−45°) … 2.0 ms (+45°)
    us = 1500 + int(deg / 45.0 * 500)
    pi.set_servo_pulsewidth(TILT_PIN, max(900, min(2100, us)))

def on_paint(msg: Int32):
    pi.gpio_write(PAINT_PIN, 1)
    time.sleep(msg.data / 1000.0)
    pi.gpio_write(PAINT_PIN, 0)

n.subscribe(Double, "/mantis/pan_cmd",       on_pan)
n.subscribe(Double, "/mantis/tilt_cmd",      on_tilt)
n.subscribe(Int32,  "/mantis/paint_trigger", on_paint)
while True: time.sleep(1)
```

Wire the same way for ROS 2 with `ros_gz_bridge` if the rig speaks ROS.

### Headless / autonomous mode

Drop the Web UI entirely and let the Pi run the loop:

```bash
python3 web_app.py --headless --auto
```

- `--headless`: no Flask, just the camera → detect → track → publish loop.
- `--auto`: boots with **Auto Serial Tracker** + **Auto Paint** enabled, so the turret begins paint-marking detected targets without any user input.

### Health endpoint for a hardware watchdog

```bash
curl http://127.0.0.1:5055/api/health
```

Returns `200` + `{"ok": true, "camera_age_s", "joint_age_s", ...}`
when the loop is alive, or `503` with a list of `issues` if camera or
joint feedback has stalled. Wire your actuator's enable line to a
watchdog that polls this once a second.

## Safe scope

Keep this project focused on:
- robotic perception
- camera simulation
- multi-object tracking
- pan/tilt servo control
- evaluation metrics
- non-physical virtual marking

Do not add:
- real firing, impact or projectile models
- instructions for building or deploying a physical launcher
- autonomous engagement logic outside this closed educational simulation
