# Mechanism And 3D Assets

This document explains how the current MANTIS PAINTER simulation makes the MANTIS body rotate and the nose tilt.

## Source Files

Original user assets:

- STL reference: `/home/sanju/Downloads/MANTIS.stl`.
- Blender source: `/home/sanju/Documents/BLENDER/MANTIS.blend`.

Current project copies and exports:

- Original STL copy: `models/mantis/meshes/MANTIS.stl`.
- Blender preview export: `models/mantis/blend_export/mantis_rig.glb`.
- Gazebo robot SDF: `models/mantis_robot/model.sdf`.
- Gazebo robot model config: `models/mantis_robot/model.config`.
- Base mesh: `models/mantis_robot/meshes/base_link.stl`.
- Pan body mesh: `models/mantis_robot/meshes/pan_link.stl`.
- Tilt nose mesh: `models/mantis_robot/meshes/tilt_link.stl`.
- Main world: `worlds/mantis_robot_world.sdf`.

The currently running 3D file in Gazebo is:

```text
/home/sanju/MANTIS_PAINTER/models/mantis_robot/model.sdf
```

It is loaded by:

```xml
<include>
  <name>mantis_robot</name>
  <pose>0 -8 0.0 0 0 0</pose>
  <uri>/home/sanju/MANTIS_PAINTER/models/mantis_robot</uri>
</include>
```

inside:

```text
/home/sanju/MANTIS_PAINTER/worlds/mantis_robot_world.sdf
```

## Blender Hierarchy Used

The Blender file contained three useful mesh objects:

- `BAREL_standardSurface1_0`: base object.
- `BAREL_standardSurface1_0.001`: pan/yaw body object.
- `BAREL_standardSurface1_0.002`: tilt/nose object.

The rotation limits found in Blender were:

- Pan object: Z rotation from `-85.3` to `+89.2` degrees.
- Tilt object: Y rotation from `-40.0` to `+30.0` degrees.

The export script that reads this is:

```text
scripts/export_mantis_robot.py
```

Run it with:

```bash
cd /home/sanju/MANTIS_PAINTER
blender --background /home/sanju/Documents/BLENDER/MANTIS.blend --python scripts/export_mantis_robot.py
```

## Gazebo Link Structure

The Gazebo robot is split into three links:

1. `base_link`
   - Fixed lower support.
   - Holds the platform in the world.

2. `pan_link`
   - Child of `base_link`.
   - Rotates around the vertical Z axis.
   - Uses `models/mantis_robot/meshes/pan_link.stl`.

3. `tilt_link`
   - Child of `pan_link`.
   - Rotates around the local Y axis.
   - Uses `models/mantis_robot/meshes/tilt_link.stl`.
   - Contains the nose camera sensor.

## Pan Joint

The pan joint is defined in `models/mantis_robot/model.sdf`:

```xml
<joint name="pan_joint" type="revolute">
  <parent>base_link</parent>
  <child>pan_link</child>
  <pose relative_to="base_link">2.259159 -2.233988 2.039962 0 0 0</pose>
  <axis>
    <xyz expressed_in="base_link">0 0 1</xyz>
    <limit>
      <lower>-1.48876584</lower>
      <upper>1.55683362</upper>
      <effort>500</effort>
      <velocity>1.2</velocity>
    </limit>
  </axis>
</joint>
```

Meaning:

- The joint rotates around Z.
- Lower limit is about `-85.3 deg`.
- Upper limit is about `+89.2 deg`.
- This is what makes the body spin left/right.

## Tilt Joint

The tilt joint is defined in `models/mantis_robot/model.sdf`:

```xml
<joint name="tilt_joint" type="revolute">
  <parent>pan_link</parent>
  <child>tilt_link</child>
  <pose relative_to="pan_link">1.739549 0.087656 1.489971 0 0 0</pose>
  <axis>
    <xyz expressed_in="pan_link">0 1 0</xyz>
    <limit>
      <lower>-0.69813168</lower>
      <upper>0.52359879</upper>
      <effort>220</effort>
      <velocity>1.0</velocity>
    </limit>
  </axis>
</joint>
```

Meaning:

- The joint rotates around local Y.
- Lower limit is about `-40 deg`.
- Upper limit is about `+30 deg`.
- This is what makes the front nose move up/down.

## Joint Controllers

Gazebo moves the joints using built-in joint position controller plugins. The
old `p_gain=8, d_gain=0.25` values were far below critical damping for the link
inertias and let the pan joint drift / sweep freely even when `data: 0` was
commanded. The current PID gains are sized so the link reaches the commanded
angle quickly and holds it under the gravity load of the offset tilt mass:

```xml
<plugin filename="gz-sim-joint-position-controller-system" name="gz::sim::systems::JointPositionController">
  <joint_name>pan_joint</joint_name>
  <topic>/mantis/pan_cmd</topic>
  <p_gain>900.0</p_gain>
  <i_gain>30.0</i_gain>
  <d_gain>500.0</d_gain>
  <i_max>400.0</i_max>
  <i_min>-400.0</i_min>
  <cmd_max>2000.0</cmd_max>
  <cmd_min>-2000.0</cmd_min>
  <initial_position>0.0</initial_position>
</plugin>
```

```xml
<plugin filename="gz-sim-joint-position-controller-system" name="gz::sim::systems::JointPositionController">
  <joint_name>tilt_joint</joint_name>
  <topic>/mantis/tilt_cmd</topic>
  <p_gain>700.0</p_gain>
  <i_gain>25.0</i_gain>
  <d_gain>360.0</d_gain>
  <i_max>300.0</i_max>
  <i_min>-300.0</i_min>
  <cmd_max>1500.0</cmd_max>
  <cmd_min>-1500.0</cmd_min>
  <initial_position>0.20943951</initial_position>
</plugin>
```

Tilt effort/velocity limits were also bumped (`<effort>1500</effort>`,
`<velocity>1.3</velocity>`) because the offset tilt-link CoM produces a gravity
torque on the tilt axis that exceeded the previous 220 Nm effort limit.

Joint `<dynamics>` damping was increased (pan 40, tilt 25) and the tilt link
inertial `<pose>` was set to the CoM offset (3 0 0.6) so the simulated dynamics
match the asymmetric mass distribution.

These plugins listen for angle commands:

- `/mantis/pan_cmd`
- `/mantis/tilt_cmd`

The command message type is:

```text
gz.msgs.Double
```

The value is in radians.

Manual examples:

```bash
gz topic -t /mantis/pan_cmd -m gz.msgs.Double -p 'data: 0.0'
gz topic -t /mantis/tilt_cmd -m gz.msgs.Double -p 'data: 0.20943951'
```

The second command sets tilt to about `+12 deg`.

## Nose Camera

The real Gazebo camera is attached to `tilt_link`:

```xml
<sensor name="nose_camera" type="camera">
  <pose>6.250000 0.000000 1.650000 0 0 0</pose>
  <topic>/mantis/nose_camera/image</topic>
  <update_rate>30</update_rate>
  <camera>
    <horizontal_fov>1.012</horizontal_fov>
    <image>
      <width>1280</width>
      <height>720</height>
      <format>R8G8B8</format>
    </image>
  </camera>
</sensor>
```

Because the camera is inside `tilt_link`, it moves with both:

- `pan_joint`
- `tilt_joint`

That is why the Web UI camera view changes when the MANTIS rotates.

## Web Controller

The Web UI controller is in:

```text
web_app.py
```

It subscribes to:

```text
/mantis/nose_camera/image
```

It publishes to:

```text
/mantis/pan_cmd
/mantis/tilt_cmd
```

The publish function converts degrees to radians:

```python
def publish_angle(pub, deg: float) -> None:
    msg = Double()
    msg.data = math.radians(deg)
    pub.publish(msg)
```

### Control modes

`mode` is one of:

- `auto` — track the selected detection toward the camera center.
- `manual` — ignore detections, follow jog targets from the UI / keyboard.
- `home` — drive smoothly to `HOME_PAN_DEG = 0`, `HOME_TILT_DEG = 12`.

### Auto-track PID

The auto loop runs inside the camera image callback at the real camera rate
(typically 30 Hz) and uses the actual frame `dt`, not a fixed `CONTROL_HZ`.

1. Pick the target via `resolve_selected_target()`. With multiple same-color
   detections, it picks the one whose bbox center is closest to the previously
   anchored `(cx, cy)` so the track does not jump between same-colored objects.
2. Normalize the bbox center to `nx, ny ∈ [-1, 1]` relative to image center.
3. Apply a deadband (`deadband_norm` in pan/tilt gains).
4. Map normalized error to a real angle in degrees using the camera FOV:

   ```python
   pan_err_deg = degrees(atan(tan(HFOV/2) * nx))
   tilt_err_deg = degrees(atan(tan(VFOV/2) * ny))
   ```

5. Compute PID terms with real `dt`. Anti-windup clamps the integral.
6. Output `u_deg = Kp*err + Ki*∫err + Kd*derr/dt`.
7. Apply sign convention `PAN_SIGN = -1`, `TILT_SIGN = +1` (verified against
   the joint axes: pan is +Z body yaw, tilt is +Y nose-down).
8. Clamp step to `max_rate_deg_s * dt`, then add to `pan_deg` / `tilt_deg` and
   publish.

### Lost-target behavior

If the selected target is not detected in the current frame the controller
holds position and decays its integral. After `LOST_GRACE_S = 0.8` seconds with
no detection it clears the selection. With no selection the controller drives
smoothly to the home pose at `HOME_MAX_RATE_DEG_S`.

### Web UI controls

- Mode buttons: Auto-track / Manual / Home / Clear target.
- Jog pad with step-size selector (0.5 – 10°) for pan/tilt.
- Keyboard: arrows or WASD jog, space = home, C = clear, M = manual, T = auto.
- Live PID sliders for `Kp`, `Ki`, `Kd`, `max_rate`, `deadband`. Pan and tilt
  share these so tuning is symmetric; tilt automatically derates `max_rate` and
  bumps deadband slightly.
- REST endpoints: `/api/mode`, `/api/jog`, `/api/gains`, `/api/select`,
  `/api/select_detection`, `/api/status`.

## Current Safety Behavior

The project uses only:

- camera feed
- bounding boxes
- virtual target selection
- pan/tilt centering
- non-physical `virtual_mark` events

It does not include projectile physics, impact simulation, or real deployment instructions.
