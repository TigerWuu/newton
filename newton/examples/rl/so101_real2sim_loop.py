# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Iterated system-identification / policy-optimization loop for the SO-101.

Runs the real-to-sim-to-real loop (SimOpt-style, cf. Chebotar et al. 2019)
by orchestrating the existing examples, one round being:

    1. identify parameters        diffsim_so101_sysid_true  (on the previous
                                  round's policy rollouts, warm-started from
                                  the previous parameter estimate)
    2. train / fine-tune policy   robot_so101_reach_shac --train
                                  (warm-started from the previous checkpoint)
    3. deploy the policy on the "real" arm and record (cmd, pos) rollouts
    4. evaluate the sim-to-real gap (open-loop replay RMSE) and stop when
       it plateaus

The "real" arm is either:

* ``--hardware``: a physical SO-101 driven closed-loop through the lerobot
  bridge (the policy reads *measured* joint states), or
* the default *virtual real arm*: a second Newton simulation whose parameters
  are the ground-truth CSV given by ``--gt-params`` (defaults to the newest
  ``logs/sysid/*/identified_parameters.csv``). This validates the whole loop
  without hardware: the loop starts from nominal CAD parameters and must
  rediscover the ground truth from its own policy rollouts.

Round 1 has no rollout data yet, so it trains on nominal parameters unless
``--init-data <recording.csv>`` provides a bootstrap recording (e.g. a chirp
run from record_tra.sh) to identify against first.

Rollouts are recorded in the same CSV format as record_tra.sh
(``t,freq_hz,<servo>_cmd,<servo>_pos`` in degrees at 50 Hz), so the sysid
example consumes them unchanged. Goals are resampled every ``--goal-period-s``
without resetting the arm, so a recording is one continuous trajectory whose
goal-switch transients provide the excitation for identification.

Usage::

    # validate the loop without hardware (virtual real arm = identified CSV)
    uv run -m newton.examples.rl.so101_real2sim_loop --rounds 3

    # on real hardware, bootstrapping from an existing chirp recording
    uv run -m newton.examples.rl.so101_real2sim_loop --hardware \\
        --init-data <path/to/chirp_06_all_joints.csv> --robot-port /dev/followerarm-right
"""

import argparse
import csv
import glob
import math
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch
import warp as wp

import newton
import newton.examples

# recorded servo channel <-> USD joint mapping, shared with the sysid example
from newton.examples.diffsim.example_diffsim_so101_sysid_true import CSV_JOINT_TO_USD
from newton.examples.rl.example_robot_so101_reach import SO101ReachEnv
from newton.examples.rl.shac import load_shac_actor
from newton.examples.rl.so101_rl_env import build_so101_world
from newton.examples.robot.example_robot_so101_digital_twin import (
    USD_JOINT_TO_MOTOR,
    SO101Hardware,
    friction_torque_kernel,
)

FPS = 50  # control/recording rate, matching record_tra.sh and the envs


def env_args(params: str | None, seed: int = 42, armature: float = 0.1, substeps: int = 4) -> argparse.Namespace:
    """Minimal argument namespace accepted by the SO-101 env builders."""
    return argparse.Namespace(
        params=params or "nominal",  # non-existent path -> nominal CAD parameters
        plot_dir="logs/sysid",
        armature=armature,
        substeps=substeps,
        seed=seed,
    )


def probe_joint_channels(args: argparse.Namespace) -> dict[str, int]:
    """Map each recorded servo channel to its joint coordinate index (one world)."""
    probe = newton.ModelBuilder()
    build_so101_world(probe, args)
    name_to_coord = {label.rsplit("/", 1)[-1]: probe.joint_q_start[j] for j, label in enumerate(probe.joint_label)}
    return {servo: name_to_coord[usd] for servo, usd in CSV_JOINT_TO_USD.items()}


def write_rollout_csv(path: str, t: np.ndarray, cmd_deg: dict, pos_deg: dict):
    """Write a recording in the record_tra.sh CSV format the sysid example reads."""
    servos = list(CSV_JOINT_TO_USD)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "freq_hz"] + [f"{s}_{k}" for s in servos for k in ("cmd", "pos")])
        for i in range(len(t)):
            row = [f"{t[i]:.6f}", "0.0"]
            for s in servos:
                row += [f"{cmd_deg[s][i]:.6f}", f"{pos_deg[s][i]:.6f}"]
            writer.writerow(row)


# --------------------------------------------------------------------------
# rollout recording (policy driving the "real" arm)
# --------------------------------------------------------------------------


def record_virtual(loop_args, actor, out_csv: str):
    """Record policy rollouts on the virtual real arm (ground-truth-parameter sim)."""
    args = env_args(loop_args.gt_params, seed=loop_args.seed)
    channels = probe_joint_channels(args)
    env = SO101ReachEnv(args, 1)
    env.max_episode_length = 1 << 30  # never auto-reset: one continuous recording

    frames = int(loop_args.rollout_seconds * FPS)
    goal_period = max(1, int(loop_args.goal_period_s * FPS))
    servos = list(channels)
    t = np.arange(frames) / FPS
    cmd_deg = {s: np.zeros(frames) for s in servos}
    pos_deg = {s: np.zeros(frames) for s in servos}

    ids = torch.tensor([0], device=env.device)
    with torch.inference_mode():
        for i in range(frames):
            if i % goal_period == 0:
                env._reset_task(ids)  # resample the goal only; the arm keeps moving
            obs = env.get_observations()
            actions = actor(obs["policy"], stochastic=False)
            env.step(actions)
            q = env.q[0]
            target = env.joint_target[0]
            for s in servos:
                c = channels[s]
                cmd_deg[s][i] = math.degrees(float(target[c]))
                pos_deg[s][i] = math.degrees(float(q[c]))

    write_rollout_csv(out_csv, t, cmd_deg, pos_deg)
    print(f"recorded {frames} frames (virtual real arm) -> {out_csv}")


def record_hardware(loop_args, actor, out_csv: str):
    """Record policy rollouts on the physical SO-101 (closed loop on measured state).

    The policy reads the *measured* joint positions (velocities by finite
    difference), exactly as it would in deployment; the commanded targets and
    measured positions are logged for identification.
    """
    args = env_args(loop_args.params_for_obs, seed=loop_args.seed)
    channels = probe_joint_channels(args)
    # env used only for the observation frame (home pose, goal region, limits)
    env = SO101ReachEnv(args, 1)
    env.max_episode_length = 1 << 30
    servos = list(channels)
    motor_of = {s: USD_JOINT_TO_MOTOR[CSV_JOINT_TO_USD[s]] for s in servos}

    hardware = SO101Hardware(loop_args.robot_port, loop_args.robot_id, FPS)
    hardware.connect()
    print(f"connected to SO-101 on {loop_args.robot_port}")

    # ease the arm into the home pose before slaving it to the policy
    home_cmd = {f"{motor_of[s]}.pos": math.degrees(float(env.default_q[channels[s]])) for s in servos}
    hardware.move_to(home_cmd, 4.0)

    frames = int(loop_args.rollout_seconds * FPS)
    goal_period = max(1, int(loop_args.goal_period_s * FPS))
    t = np.arange(frames) / FPS
    cmd_deg = {s: np.zeros(frames) for s in servos}
    pos_deg = {s: np.zeros(frames) for s in servos}

    dofs = env.num_actions
    q = env.default_q.clone()
    qd = torch.zeros(dofs, device=env.device)
    prev_action = torch.zeros(1, dofs, device=env.device)
    ids = torch.tensor([0], device=env.device)
    import time  # noqa: PLC0415

    next_tick = time.perf_counter()
    with torch.inference_mode():
        for i in range(frames):
            if i % goal_period == 0:
                env._reset_task(ids)

            # measured state -> observation (same layout as training)
            meas = hardware.read_arm_deg(list(motor_of.values()))
            q_prev = q.clone()
            for s in servos:
                q[channels[s]] = math.radians(meas[motor_of[s]])
            qd = (q - q_prev) * FPS
            obs = torch.cat([q - env.default_q, qd, (env.goal_pos - env.home_ee)[0], prev_action[0]]).unsqueeze(0)

            action = actor(obs, stochastic=False)
            target = torch.clamp(
                env.default_q + SO101ReachEnv.action_scale * action[0], env.joint_limit_lower, env.joint_limit_upper
            )
            command = {}
            for s in servos:
                deg = math.degrees(float(target[channels[s]]))
                command[f"{motor_of[s]}.pos"] = deg
                cmd_deg[s][i] = deg
                pos_deg[s][i] = meas[motor_of[s]]
            prev_action = action

            SO101Hardware._sleep_until(next_tick)
            next_tick = time.perf_counter() + 1.0 / FPS
            hardware.send(command)

    hardware.disconnect()
    write_rollout_csv(out_csv, t, cmd_deg, pos_deg)
    print(f"recorded {frames} frames (hardware) -> {out_csv}")


# --------------------------------------------------------------------------
# sim-to-real gap metric
# --------------------------------------------------------------------------


def evaluate_gap(rollout_csv: str, params_csv: str | None, loop_args) -> float:
    """Open-loop replay RMSE [rad]: drive the identified sim with the recorded
    commands and compare against the recorded (real) joint positions."""
    args = env_args(params_csv, seed=loop_args.seed)
    channels = probe_joint_channels(args)
    servos = list(channels)

    data = np.genfromtxt(rollout_csv, delimiter=",", names=True)
    frames = len(data["t"])
    cmd_rad = {s: np.deg2rad(np.asarray(data[f"{s}_cmd"])) for s in servos}
    pos_rad = {s: np.deg2rad(np.asarray(data[f"{s}_pos"])) for s in servos}

    builder = newton.ModelBuilder()
    build_so101_world(builder, args)
    model = builder.finalize()
    solver = newton.solvers.SolverFeatherstone(model)
    state_0, state_1 = model.state(), model.state()
    control = model.control()

    # start the replay from the first measured pose
    q0 = model.joint_q.numpy()
    for s in servos:
        q0[channels[s]] = pos_rad[s][0]
    state_0.joint_q.assign(q0)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    substeps = args.substeps
    sim_dt = 1.0 / (FPS * substeps)
    target = model.joint_q.numpy().copy()
    err_sq, count = 0.0, 0
    for i in range(frames):
        for s in servos:
            target[channels[s]] = cmd_rad[s][i]
        control.joint_target_q.assign(target)
        for _ in range(substeps):
            wp.launch(
                friction_torque_kernel,
                dim=model.joint_dof_count,
                inputs=[state_0.joint_qd, model.joint_friction, 0.1],
                outputs=[control.joint_f],
            )
            solver.step(state_0, state_1, control, None, sim_dt)
            state_0, state_1 = state_1, state_0
        q = state_0.joint_q.numpy()
        for s in servos:
            err_sq += float(q[channels[s]] - pos_rad[s][i]) ** 2
            count += 1
    return math.sqrt(err_sq / count)


# --------------------------------------------------------------------------
# stage runners (subprocesses over the existing examples)
# --------------------------------------------------------------------------


def run_stage(cmd: list[str], log_path: str):
    print(f"  $ {' '.join(cmd)}")
    with open(log_path, "w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        tail = open(log_path).readlines()[-25:]
        raise RuntimeError(f"stage failed (see {log_path}):\n{''.join(tail)}")


def run_sysid(loop_args, data_csv: str, init_params: str | None, round_dir: str) -> str:
    out_dir = os.path.join(round_dir, "sysid")
    cmd = [
        sys.executable, "-m", "newton.examples", "diffsim_so101_sysid_true",
        "--viewer", "null", "--quiet",
        "--data", data_csv,
        "--train-iters", str(loop_args.sysid_iters),
        # the example consumes ~2 viewer frames per optimizer iteration; extra
        # frames after finalize() are no-ops, so overshoot generously
        "--num-frames", str(2 * loop_args.sysid_iters + 10),
        "--fit-fraction", "1.0",  # policy rollouts have no chirp frequency ramp
        "--plot-dir", out_dir,
    ]  # fmt: skip
    if init_params:
        cmd += ["--resume", init_params]
    run_stage(cmd, os.path.join(round_dir, "sysid.log"))
    matches = glob.glob(os.path.join(out_dir, "*", "identified_parameters.csv"))
    if not matches:
        raise RuntimeError(f"sysid produced no identified_parameters.csv under {out_dir}")
    return max(matches, key=os.path.getmtime)


def run_shac(loop_args, params: str | None, checkpoint: str | None, iters: int, round_dir: str) -> str:
    logdir = os.path.join(round_dir, "shac")
    cmd = [
        sys.executable, "-m", "newton.examples", "robot_so101_reach_shac",
        "--train", "--quiet",
        "--num-envs", str(loop_args.num_envs),
        "--max-iterations", str(iters),
        "--seed", str(loop_args.seed),
        "--params", params or "nominal",
        "--logdir", logdir,
    ]  # fmt: skip
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    run_stage(cmd, os.path.join(round_dir, "shac.log"))
    matches = glob.glob(os.path.join(logdir, "*", "*", "model_*.pt"))
    if not matches:
        raise RuntimeError(f"SHAC produced no checkpoint under {logdir}")
    return max(matches, key=os.path.getmtime)


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=4, help="Maximum identification/training rounds.")
    parser.add_argument(
        "--gt-params",
        type=str,
        default=None,
        help="Ground-truth parameter CSV for the virtual real arm (default: newest logs/sysid/*/identified_parameters.csv).",
    )
    parser.add_argument(
        "--hardware",
        action="store_true",
        default=False,
        help="Use the physical SO-101 (lerobot) instead of the virtual real arm.",
    )
    parser.add_argument("--robot-port", type=str, default=os.environ.get("ROBOT_PORT", "/dev/followerarm-right"))
    parser.add_argument("--robot-id", type=str, default=os.environ.get("ROBOT_ID", "my_awesome_follower_arm"))
    parser.add_argument(
        "--init-data",
        type=str,
        default=None,
        help="Optional bootstrap recording (e.g. chirp CSV) to identify against in round 1.",
    )
    parser.add_argument("--rollout-seconds", type=float, default=20.0, help="Recording length per round [s].")
    parser.add_argument("--goal-period-s", type=float, default=2.5, help="Goal resampling period [s].")
    parser.add_argument("--sysid-iters", type=int, default=250)
    parser.add_argument("--shac-first-iters", type=int, default=800, help="SHAC iterations in round 1.")
    parser.add_argument("--shac-iters", type=int, default=200, help="SHAC fine-tune iterations per later round.")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument(
        "--plateau-tol", type=float, default=0.03, help="Stop when the relative gap improvement falls below this."
    )
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    loop_args = parser.parse_args()

    if not loop_args.hardware:
        from newton.examples.robot.example_robot_so101_digital_twin import find_latest_identified_csv  # noqa: PLC0415

        loop_args.gt_params = loop_args.gt_params or find_latest_identified_csv("logs/sysid")
        if not loop_args.gt_params:
            raise SystemExit("no --gt-params and no logs/sysid/*/identified_parameters.csv found for the virtual real arm")
        print(f"virtual real arm ground truth: {loop_args.gt_params}")

    workdir = loop_args.workdir or os.path.join("logs", "real2sim", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(workdir, exist_ok=True)
    print(f"workdir: {workdir}")

    params: str | None = None  # round 1 trains on nominal unless --init-data identifies first
    checkpoint: str | None = None
    rollout_csv: str | None = None
    history: list[tuple[int, str | None, float]] = []

    for r in range(1, loop_args.rounds + 1):
        round_dir = os.path.join(workdir, f"round_{r}")
        os.makedirs(round_dir, exist_ok=True)
        print(f"\n=== round {r}/{loop_args.rounds} ===")

        # 1. identify on the freshest data (previous round's rollouts, or the bootstrap recording)
        data = rollout_csv or (loop_args.init_data if r == 1 else None)
        if data:
            print(f"[identify] data: {data}" + (f"  (warm start: {params})" if params else ""))
            params = run_sysid(loop_args, data, params, round_dir)
            print(f"[identify] -> {params}")
        else:
            print("[identify] no data yet; using nominal parameters")

        # 2. train / fine-tune the policy on the identified simulation
        iters = loop_args.shac_first_iters if checkpoint is None else loop_args.shac_iters
        print(f"[train] SHAC {iters} iterations" + (f"  (warm start: {checkpoint})" if checkpoint else ""))
        checkpoint = run_shac(loop_args, params, checkpoint, iters, round_dir)
        print(f"[train] -> {checkpoint}")

        # 3. deploy the policy on the real arm and record rollouts
        actor = load_shac_actor(checkpoint, "cuda" if wp.get_device().is_cuda else "cpu")
        rollout_csv = os.path.join(round_dir, "policy_rollout.csv")
        if loop_args.hardware:
            loop_args.params_for_obs = params
            record_hardware(loop_args, actor, rollout_csv)
        else:
            record_virtual(loop_args, actor, rollout_csv)

        # 4. sim-to-real gap: replay the recorded commands through the identified sim
        gap = evaluate_gap(rollout_csv, params, loop_args)
        history.append((r, params, gap))
        print(f"[gap] open-loop replay RMSE with current parameters: {gap * 1e3:.2f} mrad")

        if len(history) >= 2 and history[-2][2] > 0:
            improvement = (history[-2][2] - gap) / history[-2][2]
            print(f"[gap] improvement over previous round: {improvement:+.1%}")
            if 0 <= improvement < loop_args.plateau_tol:
                print("[loop] gap plateaued; stopping")
                break

    print("\n=== summary ===")
    for r, p, gap in history:
        print(f"round {r}: gap {gap * 1e3:8.2f} mrad   params: {p or 'nominal'}")
    print(f"final policy: {checkpoint}")
    print(f"final parameters: {params or 'nominal'}")


if __name__ == "__main__":
    main()
