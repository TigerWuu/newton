#!/usr/bin/env bash
# =============================================================================
# record_tra.sh
#
# Records CHIRP (frequency-sweep) trajectories on a single SO-101 follower arm
# for system identification, then saves each run as .npy AND .csv.
#
# What it does:
#   0. Moves the arm to a fixed INITIAL STATE (INIT_STATE) that the chirp is
#      centered on (all 6 motors, including the gripper).
#   1. Sweeps each joint INDIVIDUALLY with a 0.1 -> 2 Hz linear chirp, in
#      REVERSE order of:  rotation, pitch, elbow, wrist_pitch, wrist_roll
#        -> wrist_roll, wrist_pitch (wrist_flex), elbow (elbow_flex),
#           pitch (shoulder_lift), rotation (shoulder_pan)
#      While one joint is swept, the others are actively held at the home pose.
#   2. Finally sweeps ALL joints AT ONCE with the SAME chirp signal.
#
# Reference: SO100/teleop.sh (robot.type / robot.port / robot.id) and the
# lerobot-record entry point (lerobot.scripts.lerobot_record:main). lerobot-record
# only *captures* a teleop/policy stream, so it cannot synthesize a chirp; this
# script instead drives the chirp through the very same lerobot robot classes
# that lerobot-record uses internally. It does NOT modify any existing code.
#
# Positions are in DEGREES (use_degrees=True). 0 deg = middle of each joint's
# calibrated range. The chirp amplitude is automatically clamped to stay inside
# each joint's safe range, so the arm cannot be driven past its limits.
#
# NOTE: this moves real hardware. Keep the workspace clear and a hand near the
# power switch. Press Ctrl-C at any time to stop (torque is released on exit).
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------- configuration ---------------------------------
# Robot (mirrors SO100/teleop.sh; calibration id must exist under
#   ~/.cache/huggingface/lerobot/calibration/robots/so_follower/<ROBOT_ID>.json)
export ROBOT_PORT="${ROBOT_PORT:-/dev/followerarm-right}"
export ROBOT_ID="${ROBOT_ID:-my_awesome_follower_arm}"

# Chirp signal
export F_START="${F_START:-0.1}"     # start frequency [Hz]
export F_END="${F_END:-2.0}"         # end   frequency [Hz]
export DURATION="${DURATION:-20.0}"  # sweep length per run [s]
export FPS="${FPS:-50.0}"            # control / sampling rate [Hz]
export AMP_DEG="${AMP_DEG:-15.0}"    # nominal chirp amplitude [deg]
export MARGIN_DEG="${MARGIN_DEG:-5.0}"  # keep-out margin from joint limits [deg]
export TAPER="${TAPER:-0.1}"         # Tukey taper fraction (smooth start/stop)
export MOVE_SECS="${MOVE_SECS:-3.0}" # time to ease back to home between runs [s]

# Initial state the chirp is centered on — one value per motor, in order:
#   rotation, pitch, elbow, wrist_pitch, wrist_roll  [deg]  +  gripper [0-100]
export INIT_STATE="${INIT_STATE:-0.0,-20.0,25.0,90.0,0.0,30.0}"
export INIT_MOVE_SECS="${INIT_MOVE_SECS:-4.0}"  # time to ease into the initial state [s]

# Output
export OUT_DIR="${OUT_DIR:-$HERE/outputs/chirp_traj}"

# Python from the project venv (avoid `uv run`, which can resync/replace torch).
PYTHON="${PYTHON:-$HERE/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="python"

echo "=============================================================="
echo " SO-101 chirp trajectory recorder"
echo "   robot.port : $ROBOT_PORT"
echo "   robot.id   : $ROBOT_ID"
echo "   chirp      : $F_START -> $F_END Hz over ${DURATION}s @ ${FPS}Hz"
echo "   amplitude  : up to ${AMP_DEG} deg (auto-clamped to joint limits)"
echo "   init state : [${INIT_STATE}]  (rot,pitch,elbow,wrist_pitch,wrist_roll deg; gripper 0-100)"
echo "   output dir : $OUT_DIR/<timestamp>/"
echo "=============================================================="

# Run the recorder from a temp file (NOT a stdin heredoc) so that stdin stays
# attached to the terminal — otherwise input()/calibration prompts hit EOF and
# the script aborts immediately.
TMP_PY="$(mktemp "${TMPDIR:-/tmp}/record_tra.XXXXXX.py")"
trap 'rm -f "$TMP_PY"' EXIT
cat > "$TMP_PY" <<'PY'
import csv
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO101Follower

# --- config from environment -------------------------------------------------
PORT       = os.environ.get("ROBOT_PORT", "/dev/followerarm-right")
ROBOT_ID   = os.environ.get("ROBOT_ID", "my_awesome_follower_arm")
F0         = float(os.environ.get("F_START", "0.1"))
F1         = float(os.environ.get("F_END", "2.0"))
DURATION   = float(os.environ.get("DURATION", "20.0"))
FPS        = float(os.environ.get("FPS", "50.0"))
AMP_DEG    = float(os.environ.get("AMP_DEG", "15.0"))
MARGIN_DEG = float(os.environ.get("MARGIN_DEG", "5.0"))
TAPER      = float(os.environ.get("TAPER", "0.1"))
MOVE_SECS  = float(os.environ.get("MOVE_SECS", "3.0"))
INIT_MOVE_SECS = float(os.environ.get("INIT_MOVE_SECS", "4.0"))
INIT_STATE = [float(x) for x in
              os.environ.get("INIT_STATE", "0.0,-20.0,25.0,90.0,0.0,30.0").split(",")]
OUT_ROOT   = Path(os.environ.get("OUT_DIR", "outputs/chirp_traj"))

STS3215_MAX_RES = 4095  # 4096-tick encoder -> degrees = (raw-mid)*360/4095

# 5 arm joints, mapped from the user's joint names to the lerobot motor names.
USER_LABEL = {
    "shoulder_pan":  "rotation",
    "shoulder_lift": "pitch",
    "elbow_flex":    "elbow",
    "wrist_flex":    "wrist_pitch",
    "wrist_roll":    "wrist_roll",
}
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
# Reverse order of: rotation, pitch, elbow, wrist_pitch, wrist_roll
SINGLE_ORDER = ["wrist_roll", "wrist_flex", "elbow_flex", "shoulder_lift", "shoulder_pan"]
GRIPPER = "gripper"
ALL_MOTORS = ARM + [GRIPPER]  # canonical order; matches the 6 INIT_STATE values


def tukey(t: float, total: float, alpha: float) -> float:
    """Tapered window in [0,1]: smooth ramp in/out so motion starts/ends at home."""
    if alpha <= 0.0:
        return 1.0
    edge = alpha * total / 2.0
    if t < edge:
        return 0.5 * (1.0 - math.cos(math.pi * t / edge))
    if t > total - edge:
        return 0.5 * (1.0 - math.cos(math.pi * (total - t) / edge))
    return 1.0


def read_arm(robot) -> dict[str, float]:
    obs = robot.get_observation()
    return {j: float(obs[f"{j}.pos"]) for j in ARM}


def move_to(robot, target: dict[str, float], joints: list[str], secs: float, fps: float) -> None:
    """Cosine-eased move from the current pose to `target` over `joints`."""
    obs = robot.get_observation()
    start = {j: float(obs[f"{j}.pos"]) for j in joints}
    n = max(1, int(round(secs * fps)))
    t0 = time.perf_counter()
    for i in range(1, n + 1):
        a = 0.5 * (1.0 - math.cos(math.pi * i / n))  # 0 -> 1, ease in/out
        cmd = {f"{j}.pos": start[j] + (target[j] - start[j]) * a for j in joints}
        robot.send_action(cmd)
        nxt = t0 + i / fps
        dt = nxt - time.perf_counter()
        if dt > 0:
            time.sleep(dt)


def run_chirp(robot, active, home, amps, limits):
    """Drive `active` joints with the chirp (others held at home); record rows."""
    cols = ["t", "freq_hz"]
    for j in ARM:
        cols += [f"{j}_cmd", f"{j}_pos"]

    n = int(round(DURATION * FPS))
    k = (F1 - F0) / DURATION  # linear chirp rate [Hz/s]
    rows = []
    t0 = time.perf_counter()
    for i in range(n + 1):
        t = i / FPS
        w = tukey(t, DURATION, TAPER)
        phase = 2.0 * math.pi * (F0 * t + 0.5 * k * t * t)
        s = math.sin(phase)
        freq = F0 + k * t

        cmd = {}
        for j in ARM:
            if j in active:
                lo, hi = limits[j]
                val = home[j] + amps[j] * w * s
                val = min(hi - MARGIN_DEG, max(lo + MARGIN_DEG, val))
            else:
                val = home[j]
            cmd[j] = val

        robot.send_action({f"{j}.pos": cmd[j] for j in ARM})
        meas = read_arm(robot)

        row = [t, freq]
        for j in ARM:
            row += [cmd[j], meas[j]]
        rows.append(tuple(row))

        nxt = t0 + (i + 1) / FPS
        dt = nxt - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
    return cols, rows


def save(out_dir: Path, label: str, cols, rows) -> tuple[Path, Path]:
    npy_path = out_dir / f"chirp_{label}.npy"
    csv_path = out_dir / f"chirp_{label}.csv"
    arr = np.array(rows, dtype=[(c, "f8") for c in cols])
    np.save(npy_path, arr)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    return npy_path, csv_path


def main() -> int:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SOFollowerRobotConfig(port=PORT, id=ROBOT_ID, use_degrees=True)
    robot = SO101Follower(cfg)

    # Safe per-joint degree limits from calibration: deg in [-H, +H], H from range.
    limits = {}
    for j in ARM:
        cal = robot.calibration[j]
        H = (cal.range_max - cal.range_min) / 2.0 * 360.0 / STS3215_MAX_RES
        limits[j] = (-H, +H)

    print(f"\nArm will move. Chirp {F0}->{F1} Hz, {DURATION}s/run @ {FPS}Hz.")
    print(f"Saving to: {out_dir}")
    try:
        input("Press ENTER to connect and start (Ctrl-C to abort)... ")
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        return 1

    robot.connect()  # may prompt once to write the existing calibration file
    try:
        # Desired initial state for all 6 motors (deg for arm, 0-100 for gripper).
        if len(INIT_STATE) != len(ALL_MOTORS):
            raise SystemExit(
                f"INIT_STATE needs {len(ALL_MOTORS)} values "
                f"({', '.join(ALL_MOTORS)}), got {len(INIT_STATE)}"
            )
        init_pose = dict(zip(ALL_MOTORS, INIT_STATE))
        for j in ARM:  # clamp into the safe joint range before moving there
            lo, hi = limits[j]
            c = min(hi - MARGIN_DEG, max(lo + MARGIN_DEG, init_pose[j]))
            if abs(c - init_pose[j]) > 1e-6:
                print(f"  ! init {USER_LABEL[j]} clamped {init_pose[j]:+.1f} -> {c:+.1f} deg")
            init_pose[j] = c
        init_pose[GRIPPER] = min(100.0, max(0.0, init_pose[GRIPPER]))

        print("\nMoving to initial state:")
        for j in ALL_MOTORS:
            unit = "" if j == GRIPPER else " deg"
            print(f"  {USER_LABEL.get(j, j):<12} ({j:<13}) = {init_pose[j]:+7.2f}{unit}")
        move_to(robot, init_pose, ALL_MOTORS, INIT_MOVE_SECS, FPS)

        # Chirp is centered on the initial state (5 arm joints; gripper held fixed).
        home = {j: init_pose[j] for j in ARM}

        # Auto-clamp amplitude so home +/- amp stays inside [lo+margin, hi-margin].
        amps = {}
        for j in ARM:
            lo, hi = limits[j]
            amp = min(AMP_DEG, (home[j] - lo) - MARGIN_DEG, (hi - home[j]) - MARGIN_DEG)
            amps[j] = max(0.0, amp)
            if amps[j] < AMP_DEG - 1e-6:
                print(f"  ! {USER_LABEL[j]} amplitude reduced to {amps[j]:.1f} deg "
                      f"(near joint limit)")

        # Build run list: each joint alone (reverse order), then all at once.
        runs = [(f"{i + 1:02d}_{j}", [j]) for i, j in enumerate(SINGLE_ORDER)]
        runs.append((f"{len(SINGLE_ORDER) + 1:02d}_all_joints", list(ARM)))

        for idx, (label, active) in enumerate(runs, start=1):
            names = ", ".join(USER_LABEL[j] for j in active)
            print(f"\n[{idx}/{len(runs)}] chirp: {names}")
            move_to(robot, home, ARM, MOVE_SECS, FPS)  # back to initial state first
            cols, rows = run_chirp(robot, active, home, amps, limits)
            npy_path, csv_path = save(out_dir, label, cols, rows)
            print(f"    saved {npy_path.name} and {csv_path.name} "
                  f"({len(rows)} samples)")

        print("\nReturning to initial state...")
        move_to(robot, home, ARM, MOVE_SECS, FPS)
        print(f"\nDone. All trajectories saved in:\n  {out_dir}")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted — stopping.")
        return 1
    finally:
        robot.disconnect()  # releases torque


if __name__ == "__main__":
    sys.exit(main())
PY

"$PYTHON" "$TMP_PY" "$@"
