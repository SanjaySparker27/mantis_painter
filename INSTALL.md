# Install — clone + run

Tested on Ubuntu 24.04 + Gazebo Harmonic. macOS works with **webcam mode**
(no Gazebo). Adjust commands for your distro.

## macOS quickstart (webcam-only — no Gazebo sim)

```bash
brew install python@3.12
git clone https://github.com/SanjaySparker27/mantis_painter
cd mantis_painter
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 web_app.py
```

Open http://127.0.0.1:5055, click **Connect** (top-left) → **Laptop
Webcam #0**. macOS will pop a camera permission prompt — approve it.

Skip `gz-harmonic`, the world SDF, and the Fuel model downloads on Mac.
The tracker, detector, sweep painter, web UI all work the same on top
of any webcam.

For the **YOLO-World** detector, see §2a below for CLIP weights.

---

## Linux full install (with Gazebo sim)

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

`requirements.txt` pulls a `clip` package from git for the optional
YOLO-World open-vocab detector. If your machine doesn't have git/network,
strip the last line of `requirements.txt`; the closed-set YOLOv12 detector
and HSV detector still work without CLIP.

### 2a. YOLO-World CLIP weights (first time only)

The `YOLO-World` button in the UI uses CLIP-ViT-B/32 text embeddings.
Ultralytics tries to auto-download (~354 MB) on first use; if your network
fails the SHA check, fetch manually:

```bash
mkdir -p ~/.cache/clip
curl -fL -o ~/.cache/clip/ViT-B-32.pt \
  https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
```

The web app monkey-patches CLIP's SHA check so a re-uploaded CDN copy still
loads — no further action needed.

### 2b. Webcam permission (macOS / Linux)

To use the **Connect → Laptop Webcam** option:

- **macOS**: first launch will prompt for camera permission. Approve in
  System Settings → Privacy & Security → Camera.
- **Linux**: add yourself to the `video` group:

```bash
sudo usermod -aG video $USER
# log out + back in (or reboot) for the group change to apply
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
