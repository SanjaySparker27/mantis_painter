# Install — clone + run on any Linux box

Tested on Ubuntu 24.04 + Gazebo Harmonic. Adjust commands for your distro.

## 1. System packages

```bash
sudo apt update
sudo apt install -y \
    gz-harmonic \
    python3-gz-msgs10 python3-gz-transport13 \
    python3-pip python3-venv \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
```

If `gz-harmonic` isn't found, follow the official guide:
https://gazebosim.org/docs/harmonic/install_ubuntu

## 2. Python deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Optional — local LLM agent (Ollama)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:0.5b      # small + fast, good for the agent
# or any other model you like; web_app auto-detects what's installed
```

## 4. First run

```bash
# Terminal 1 — sim
gz sim -v 3 worlds/mantis_robot_world.sdf

# Terminal 2 — web UI
python3 web_app.py
# open http://127.0.0.1:5055
```

On first run:
- `ultralytics` will download `yolo12n.pt` (~5 MB) to the repo root.
- Gazebo will fetch `prius hybrid`, `pickup` and `standing person` from
  Fuel the first time the world loads (needs internet). PX4-derived models
  (`arucotag`, `helipad`, `r1_rover`, `rc_cessna`, `x500`) are bundled in
  `models/external/` and need no download.

## 5. Headless / autonomous boot (for a Pi / on-board MCU)

```bash
python3 web_app.py --headless --auto
```

- `--headless` — run the control loop only, no Flask UI
- `--auto` — start with `Auto Serial Tracker` + `Auto Paint` enabled

External hardware can subscribe to:
- `/mantis/paint_trigger` (gz topic, `gz.msgs.Int32` = pulse width ms)
- `/tmp/mantis_paint.signal` (one line per pulse)
- UDP / TCP / serial — enable the channels you need via the Web UI or
  `POST /api/channels`.

## 6. Verify

```bash
curl -s http://127.0.0.1:5055/api/health | python3 -m json.tool
```

Expect `"ok": true` once camera + joint_states are flowing.
