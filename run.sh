#!/usr/bin/env bash
# MANTIS PAINTER — one-shot launcher. Starts Gazebo + Flask Web UI in the
# background, prints the URL. Ctrl-C to stop both.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

PY="${PYTHON:-python3}"
WORLD="${WORLD:-worlds/mantis_robot_world.sdf}"

# Make external PX4 models discoverable by Gazebo via relative paths.
export GZ_SIM_RESOURCE_PATH="$HERE/models:$HERE/models/external:${GZ_SIM_RESOURCE_PATH:-}"

cleanup() {
    echo "[mantis] shutting down..."
    [[ -n "${SIM_PID:-}" ]] && kill "$SIM_PID" 2>/dev/null || true
    [[ -n "${WEB_PID:-}" ]] && kill "$WEB_PID" 2>/dev/null || true
}
trap cleanup INT TERM

echo "[mantis] launching Gazebo with $WORLD ..."
gz sim -v 3 "$WORLD" >/tmp/mantis_gz.log 2>&1 &
SIM_PID=$!

# wait for sim topics
for i in $(seq 1 30); do
    if gz topic -l 2>/dev/null | grep -q /mantis/nose_camera/image; then
        break
    fi
    sleep 1
done

# unpause the sim
gz service -s /world/mantis_robot_world/control \
    --reqtype gz.msgs.WorldControl --reptype gz.msgs.Boolean \
    --timeout 3000 --req 'pause: false' >/dev/null 2>&1 || true

echo "[mantis] launching Web UI ..."
"$PY" web_app.py "$@" >/tmp/mantis_web.log 2>&1 &
WEB_PID=$!

sleep 4
echo "[mantis] open  http://127.0.0.1:5055"
echo "[mantis] logs   /tmp/mantis_gz.log  /tmp/mantis_web.log"
echo "[mantis] press Ctrl-C to stop"
wait
