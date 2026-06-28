# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Diffsim SO-101 System Identification (real hardware data)
#
# Identifies the inertial and joint-level parameters of a physical
# SO-ARM101 arm from chirp trajectories recorded on the real robot, by
# differentiating through the simulation with SolverFeatherstone. This
# is the hardware counterpart of ``example_diffsim_so101_sysid``, which
# fits against synthetic ground truth; here the reference comes from
# logged servo encoder readings.
#
# The recording (see ``record_tra.sh``) sweeps every joint at once with
# a 0.1 -> 2 Hz linear chirp position command at 50 Hz, logging for each
# of the five arm joints both the commanded position (``*_cmd``) and the
# measured encoder position (``*_pos``), in DEGREES, with 0 deg at each
# joint's calibrated mid-range. The gripper (Jaw) is not excited.
#
# The simulation is driven with the recorded *command* as the joint
# position target and the resulting simulated joint angles are fit to
# the recorded *measured* angles. Because the real STS3215 servos run
# their own internal position loop, the per-joint position-drive gains
# (``target_ke`` / ``target_kd``) are identified alongside the requested
# physical quantities:
#
#   - per-link mass and rotational inertia (scale factors on the nominal
#     CAD values imported from the USD file)
#   - per-joint viscous damping
#   - per-joint Coulomb (dry) friction
#   - per-joint servo position-loop stiffness and damping
#
# Short-horizon fitting: differentiating a free run over the full 20 s
# trajectory makes the adjoint explode (the stiff servo dynamics amplify
# sensitivities exponentially), and a full-trajectory graph is too large
# to capture. Instead a fixed handful of short windows, spread across the
# recording, are fit together every optimizer step. Each window is
# re-initialized from the measured joint state and rolled out for a
# fraction of a second, so the loss is the multi-step prediction error
# and the gradients stay bounded. This is the standard formulation for
# identifying dynamics from logged trajectories. The rigid-body + linear
# position-drive model is only faithful up to the drive bandwidth
# (~2 Hz on this arm), so by default the fitted windows are taken from
# the lower-frequency part of the sweep (see ``--fit-fraction``).
#
# Note on identifiability: from position-tracking data the servo
# stiffness and the link inertia enter the response mainly through their
# ratio (the drive bandwidth), so ``target_ke`` and the mass/inertia
# scales are only jointly identifiable. The damping, friction, and drive
# bandwidth are well constrained; treat the absolute mass/inertia split
# as approximate unless motor-torque measurements are also available.
#
# Command: python -m newton.examples diffsim_so101_sysid_true \
#              --data <path/to/chirp_06_all_joints.csv>
#
###########################################################################

import argparse
import csv
import os
from datetime import datetime

import numpy as np
import warp as wp
import warp.optim

import newton
import newton.examples

# Default location of the recorded chirp trajectory (the "all joints at once"
# run). Override with --data. Kept as a convenience for the author's setup; the
# example errors clearly if the file is missing.
DEFAULT_DATA = os.path.expanduser("./newton/examples/diffsim/gt/chirp_traj/20260616_133149/chirp_06_all_joints.csv")

# Base file name for the saved trajectory plots (under ``--plot-dir``).
PLOT_BASENAME = "sysid_true_trajectories"

# Mapping from the recorded servo channel (CSV column prefix) to the SO-101 USD
# joint name. The gripper (Jaw) is not part of the recording.
CSV_JOINT_TO_USD = {
    "shoulder_pan": "Rotation",
    "shoulder_lift": "Pitch",
    "elbow_flex": "Elbow",
    "wrist_flex": "Wrist_Pitch",
    "wrist_roll": "Wrist_Roll",
}

# Real-robot -> USD joint-frame calibration. The recorder logs degrees with
# 0 deg at each joint's calibrated mid-range; the USD joints are zeroed at the
# same mechanical center (their limits are symmetric about 0), so the default
# map is a pure deg->rad conversion with unit sign and no offset. Flip a sign or
# add an offset [rad] here if a channel is mirrored or shifted on your hardware.
JOINT_SIGN = dict.fromkeys(CSV_JOINT_TO_USD, +1.0)
JOINT_OFFSET_RAD = dict.fromkeys(CSV_JOINT_TO_USD, 0.0)


@wp.kernel
def copy_param_kernel(
    param: wp.array[float],
    # outputs
    model_array: wp.array[float],
):
    tid = wp.tid()
    model_array[tid] = param[tid]


@wp.kernel
def compute_spatial_inertia_kernel(
    body_mass_ref: wp.array[float],
    body_inertia_ref: wp.array[wp.mat33],
    mass_scale: wp.array[float],
    inertia_scale: wp.array[float],
    # outputs
    body_I_m: wp.array[wp.spatial_matrix],
):
    b = wp.tid()
    m = body_mass_ref[b] * mass_scale[b]
    I = body_inertia_ref[b] * inertia_scale[b]
    # fmt: off
    body_I_m[b] = wp.spatial_matrix(
        m,   0.0, 0.0, 0.0,     0.0,     0.0,
        0.0, m,   0.0, 0.0,     0.0,     0.0,
        0.0, 0.0, m,   0.0,     0.0,     0.0,
        0.0, 0.0, 0.0, I[0, 0], I[0, 1], I[0, 2],
        0.0, 0.0, 0.0, I[1, 0], I[1, 1], I[1, 2],
        0.0, 0.0, 0.0, I[2, 0], I[2, 1], I[2, 2],
    )
    # fmt: on


@wp.kernel
def set_state_from_ref_kernel(
    q_ref: wp.array2d[float],
    qd_ref: wp.array2d[float],
    starts: wp.array[int],
    slot: int,
    # outputs
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
):
    tid = wp.tid()
    f = starts[slot]
    joint_q[tid] = q_ref[f, tid]
    joint_qd[tid] = qd_ref[f, tid]


@wp.kernel
def set_target_kernel(
    cmd_ref: wp.array2d[float],
    starts: wp.array[int],
    slot: int,
    frame_in_window: int,
    # outputs
    joint_target_q: wp.array[float],
):
    tid = wp.tid()
    joint_target_q[tid] = cmd_ref[starts[slot] + frame_in_window, tid]


@wp.kernel
def friction_torque_kernel(
    joint_qd: wp.array[float],
    friction_param: wp.array[float],
    vel_eps: float,
    # outputs
    joint_f: wp.array[float],
):
    tid = wp.tid()
    # smooth Coulomb friction so the loss stays differentiable around qd = 0
    joint_f[tid] = -friction_param[tid] * wp.tanh(joint_qd[tid] / vel_eps)


@wp.kernel
def trajectory_loss_kernel(
    joint_q: wp.array[float],
    q_ref: wp.array2d[float],
    arm_dofs: wp.array[int],
    starts: wp.array[int],
    slot: int,
    frame_in_window: int,
    norm: float,
    # outputs
    loss: wp.array[float],
):
    i = wp.tid()
    dof = arm_dofs[i]
    f = starts[slot] + frame_in_window
    err = joint_q[dof] - q_ref[f, dof]
    wp.atomic_add(loss, 0, err * err * norm)


@wp.kernel
def clamp_param_kernel(
    lower: float,
    upper: float,
    # outputs
    param: wp.array[float],
):
    tid = wp.tid()
    param[tid] = wp.clamp(param[tid], lower, upper)


def load_chirp_csv(path):
    """Load a recorded chirp run.

    Returns:
        t: sample times [s], shape (T,).
        cmd_deg, pos_deg: commanded / measured joint positions [deg], each a
            dict keyed by CSV joint name with shape (T,) arrays.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"recorded trajectory not found: {path}\n"
            "Pass --data <path/to/chirp_06_all_joints.csv> pointing at a run "
            "produced by record_tra.sh."
        )
    data = np.genfromtxt(path, delimiter=",", names=True)
    t = np.asarray(data["t"], dtype=np.float64)
    cmd_deg = {j: np.asarray(data[f"{j}_cmd"], dtype=np.float64) for j in CSV_JOINT_TO_USD}
    pos_deg = {j: np.asarray(data[f"{j}_pos"], dtype=np.float64) for j in CSV_JOINT_TO_USD}
    return t, cmd_deg, pos_deg


class Example:
    def __init__(self, viewer, args):
        self.verbose = args.verbose
        self.train_iter = 0
        self.render_interval = 8

        self.viewer = viewer

        # --- load recorded trajectory ------------------------------------
        t, cmd_deg, pos_deg = load_chirp_csv(args.data)
        stride = max(int(args.stride), 1)
        sel = slice(None, args.max_frames * stride if args.max_frames > 0 else None, stride)
        t = t[sel]
        self.num_frames = len(t)
        self.fps = 1.0 / float(np.median(np.diff(t)))
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(int(args.substeps), 1)
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.window_len = max(int(args.window), 1)

        # Fit a fixed set of short windows spread evenly across the lower part of
        # the chirp, so the loss is deterministic and each window is an
        # independent short-horizon rollout (see the module docstring). The
        # rigid-body + linear position-drive model is only faithful up to the
        # drive bandwidth (~2 Hz on this arm); ``fit_fraction`` keeps the fitted
        # windows below the 2 Hz end of the sweep where that model holds.
        usable = max(int((self.num_frames - self.window_len) * args.fit_fraction), 0)
        n_windows = min(max(int(args.num_windows), 1), usable + 1)
        self.window_start_list = sorted(set(np.linspace(0, usable, n_windows).astype(int).tolist()))
        self.num_slots = len(self.window_start_list)

        # report + plot a comparison once this many optimizer steps have run,
        # and save a trajectory plot every ``plot_interval`` steps along the way
        self.target_iters = max(int(args.train_iters), 2)
        self.make_plot = args.plot
        self.animate = args.animate
        self.plot_dir = os.path.join(args.plot_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.plot_interval = max(int(args.plot_interval), 0)
        self._finalized = False
        self._idle_frame = 0
        # results (CSV, plots, animation) are written here
        os.makedirs(self.plot_dir, exist_ok=True)
        # last fitted frame, for marking the model-valid region on the plot
        self.fit_boundary = usable + self.window_len

        # initial joint position drive gains (servo loop is identified from here).
        # These are also the nominal baseline the identified values are compared
        # against (real hardware has no ground-truth parameters).
        self.init_ke = 15.0
        self.init_kd = 0.5
        # Reflected rotor inertia of the geared STS3215 servos. This dominates
        # the bare link inertia (~1e-4), sets the low observed drive bandwidth
        # (~12 rad/s, the arm barely follows the 2 Hz end of the chirp), and
        # keeps the stiff articulation numerically stable at the simulation dt.
        armature = 0.1
        self.friction_vel_eps = 0.1

        so101 = newton.ModelBuilder()
        so101.add_usd(
            newton.examples.get_asset("so101.usd"),
            enable_self_collisions=False,
            collapse_fixed_joints=False,
            hide_collision_shapes=True,
        )
        for dof in range(len(so101.joint_target_ke)):
            so101.joint_target_ke[dof] = self.init_ke
            so101.joint_target_kd[dof] = self.init_kd
            so101.joint_armature[dof] = armature

        so101.add_ground_plane()  # visual reference only; no contacts are computed

        # use `requires_grad=True` to create a model for differentiable simulation
        self.model = so101.finalize(requires_grad=True)
        self.dof_count = self.model.joint_dof_count
        assert self.dof_count == 6, "expected the 6-DoF SO-101 arm"

        # locate the DoF index of each recorded joint and the end-effector body
        qd_start = self.model.joint_qd_start.numpy()
        name_to_dof = {}
        for joint_idx, label in enumerate(self.model.joint_label):
            name_to_dof[label.rsplit("/", 1)[-1]] = int(qd_start[joint_idx])
        self.arm_dof_list = [name_to_dof[CSV_JOINT_TO_USD[j]] for j in CSV_JOINT_TO_USD]
        self.arm_dofs = wp.array(self.arm_dof_list, dtype=int)
        self.ee_body = next(i for i, label in enumerate(self.model.body_label) if label.endswith("/gripper"))

        self.solver = newton.solvers.SolverFeatherstone(self.model)

        # nominal (CAD) body properties that the scale parameters act on
        self.body_mass_ref = wp.clone(self.model.body_mass, requires_grad=False)
        self.body_inertia_ref = wp.clone(self.model.body_inertia, requires_grad=False)

        # parameters to identify
        self.ke_param = wp.full(self.dof_count, self.init_ke, dtype=float, requires_grad=True)
        self.kd_param = wp.full(self.dof_count, self.init_kd, dtype=float, requires_grad=True)
        self.damping_param = wp.zeros(self.dof_count, dtype=float, requires_grad=True)
        self.friction_param = wp.zeros(self.dof_count, dtype=float, requires_grad=True)
        self.mass_scale = wp.ones(self.model.body_count, dtype=float, requires_grad=True)
        self.inertia_scale = wp.ones(self.model.body_count, dtype=float, requires_grad=True)

        # --- map recorded data into the USD joint frame [rad] ------------
        cmd_rad = np.zeros((self.num_frames, self.dof_count), dtype=np.float32)
        pos_rad = np.zeros((self.num_frames, self.dof_count), dtype=np.float32)
        for j, dof in zip(CSV_JOINT_TO_USD, self.arm_dof_list, strict=True):
            sign, offset = JOINT_SIGN[j], JOINT_OFFSET_RAD[j]
            cmd_rad[:, dof] = sign * np.deg2rad(cmd_deg[j][sel]) + offset
            pos_rad[:, dof] = sign * np.deg2rad(pos_deg[j][sel]) + offset
        # measured joint velocities (central finite differences of the encoders)
        vel_rad = np.gradient(pos_rad, self.frame_dt, axis=0).astype(np.float32)
        self.cmd_ref = wp.array(cmd_rad, dtype=float, requires_grad=False)
        self.q_ref = wp.array(pos_rad, dtype=float, requires_grad=False)
        self.qd_ref = wp.array(vel_rad, dtype=float, requires_grad=False)
        # host copies for the open-loop replay used by the final report/plot
        self._cmd_np = cmd_rad
        self._pos_np = pos_rad
        self._vel_np = vel_rad

        # The recorded motion can exceed the USD asset's joint limits (e.g. the
        # real wrist_flex sweeps to ~105 deg while the USD Wrist_Pitch limit is
        # 95 deg). Without widening, the limit-penalty springs clamp the joint
        # and disable its drive, so it can never track the measured peaks. Relax
        # the limits to cover the recorded command/measurement range plus a
        # margin (this only ever widens, never tightens).
        margin = np.deg2rad(5.0)
        lo = self.model.joint_limit_lower.numpy()
        hi = self.model.joint_limit_upper.numpy()
        lo[: self.dof_count] = np.minimum(lo[: self.dof_count], np.minimum(cmd_rad, pos_rad).min(axis=0) - margin)
        hi[: self.dof_count] = np.maximum(hi[: self.dof_count], np.maximum(cmd_rad, pos_rad).max(axis=0) + margin)
        self.model.joint_limit_lower.assign(lo)
        self.model.joint_limit_upper.assign(hi)

        # one slot per fitted window: each slot is an independent short-horizon
        # rollout with its own state/control buffers, so backward passes do not
        # alias across windows
        self.starts = wp.array(np.array(self.window_start_list, dtype=np.int32), dtype=int)
        self.slot_states = []
        self.slot_controls = []
        for _ in range(self.num_slots):
            self.slot_states.append([self.model.state() for _ in range(self.window_len * self.sim_substeps + 1)])
            self.slot_controls.append([self.model.control() for _ in range(self.window_len * self.sim_substeps)])

        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.loss_history = []

        # forward-kinematics of the measured trajectory, for visual comparison
        self.ee_traj_ref = self.forward_kinematics_ee(pos_rad)

        self.opt_servo = warp.optim.Adam(
            [self.ke_param, self.kd_param, self.damping_param, self.friction_param], lr=5e-3
        )
        self.opt_scale = warp.optim.Adam([self.mass_scale, self.inertia_scale], lr=5e-3)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.55, 0.55, 0.4), pitch=-18.0, yaw=-135.0)

        # capture forward/backward passes
        self.capture()

    def capture(self):
        # warm up once outside the capture so the solver can allocate its
        # per-state auxiliary arrays before graph capture begins
        self.forward_backward()
        self.tape.zero()
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.forward_backward()
            self.graph = capture.graph

    def forward_backward(self):
        self.tape = wp.Tape()
        with self.tape:
            self.forward()
        self.tape.backward(self.loss)

    def forward(self):
        self.loss.zero_()
        self.rollout()
        return self.loss

    def rollout(self):
        # push the identified parameters into the model arrays read by the
        # solver, inside the tape so gradients accumulate back into them
        wp.launch(copy_param_kernel, dim=self.dof_count, inputs=[self.ke_param], outputs=[self.model.joint_target_ke])
        wp.launch(copy_param_kernel, dim=self.dof_count, inputs=[self.kd_param], outputs=[self.model.joint_target_kd])
        wp.launch(
            copy_param_kernel, dim=self.dof_count, inputs=[self.damping_param], outputs=[self.model.joint_damping]
        )
        # SolverFeatherstone bakes the spatial inertia at construction time;
        # recomputing body_I_m here makes the dynamics differentiable w.r.t. the
        # mass / inertia scales
        wp.launch(
            compute_spatial_inertia_kernel,
            dim=self.model.body_count,
            inputs=[self.body_mass_ref, self.body_inertia_ref, self.mass_scale, self.inertia_scale],
            outputs=[self.solver.body_I_m],
        )
        norm = 1.0 / (self.num_slots * self.window_len * len(self.arm_dof_list))
        for b in range(self.num_slots):
            states = self.slot_states[b]
            controls = self.slot_controls[b]
            # re-initialize the window from the measured joint state; this makes
            # states[0] a constant leaf so the adjoint does not chain across
            # windows and stays bounded
            wp.launch(
                set_state_from_ref_kernel,
                dim=self.dof_count,
                inputs=[self.q_ref, self.qd_ref, self.starts, b],
                outputs=[states[0].joint_q, states[0].joint_qd],
            )
            for i in range(self.window_len):
                for s in range(self.sim_substeps):
                    t = i * self.sim_substeps + s
                    # put target command input into the joint_target_q array
                    wp.launch(
                        set_target_kernel,
                        dim=self.dof_count,
                        inputs=[self.cmd_ref, self.starts, b, i],
                        outputs=[controls[t].joint_target_q],
                    )
                    wp.launch(
                        friction_torque_kernel,
                        dim=self.dof_count,
                        inputs=[states[t].joint_qd, self.friction_param, self.friction_vel_eps],
                        outputs=[controls[t].joint_f],
                    )
                    # run physics solver for one time step with the target command
                    self.solver.step(states[t], states[t + 1], controls[t], None, self.sim_dt)
                # compute loss for this window
                wp.launch(
                    trajectory_loss_kernel,
                    dim=len(self.arm_dof_list),
                    inputs=[
                        states[(i + 1) * self.sim_substeps].joint_q,
                        self.q_ref,
                        self.arm_dofs,
                        self.starts,
                        b,
                        i,
                        norm,
                    ],
                    outputs=[self.loss],
                )

    def step(self):
        # training is complete once finalize() has run: freeze the optimizer at
        # the identified values (the viewer keeps the window open for inspection)
        if self._finalized:
            return

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.forward_backward()

        self.loss_history.append(self.loss.numpy()[0])

        self.opt_servo.step([self.ke_param.grad, self.kd_param.grad, self.damping_param.grad, self.friction_param.grad])
        self.opt_scale.step([self.mass_scale.grad, self.inertia_scale.grad])
        self.tape.zero()

        # project the parameters back onto their physically plausible ranges
        wp.launch(clamp_param_kernel, dim=self.dof_count, inputs=[0.1, 1.0e3], outputs=[self.ke_param])
        wp.launch(clamp_param_kernel, dim=self.dof_count, inputs=[0.0, 1.0e2], outputs=[self.kd_param])
        wp.launch(clamp_param_kernel, dim=self.dof_count, inputs=[0.0, 5.0], outputs=[self.damping_param])
        wp.launch(clamp_param_kernel, dim=self.dof_count, inputs=[0.0, 5.0], outputs=[self.friction_param])
        wp.launch(clamp_param_kernel, dim=self.model.body_count, inputs=[0.1, 10.0], outputs=[self.mass_scale])
        wp.launch(clamp_param_kernel, dim=self.model.body_count, inputs=[0.1, 10.0], outputs=[self.inertia_scale])

        rmse_deg = np.rad2deg(np.sqrt(self.loss_history[-1]))
        print(f"train iter: {self.train_iter:4d} loss: {self.loss_history[-1]:.4e} (rmse {rmse_deg:.3f} deg)")

        # periodic trajectory snapshot so identification progress can be tracked
        if self.make_plot and self.plot_interval > 0 and self.train_iter % self.plot_interval == 0:
            identified_traj = self.save_trajectory_plot(f"_iter{self.train_iter:05d}")
            self._save_animation(identified_traj, os.path.join(self.plot_dir, f"identification_animation_{self.train_iter:05d}.gif"))
        self.train_iter += 1
        if self.train_iter >= self.target_iters:
            self.finalize()

    def finalize(self):
        """Print the report and save the parameter CSV, trajectory plot, and
        validation animation (once)."""
        if self._finalized:
            return
        self._finalized = True
        nominal, identified = self._parameter_sets()
        self._print_comparison(nominal, identified)
        self._save_parameters_csv(nominal, identified, os.path.join(self.plot_dir, "identified_parameters.csv"))
        if self.make_plot:
            self._plot_loss_curve(os.path.join(self.plot_dir, "loss_curve.png"))
            identified_traj = self.save_trajectory_plot(f"_iter{self.train_iter:05d}_final")
            if self.animate:
                self._save_animation(identified_traj, os.path.join(self.plot_dir, "validation_animation.gif"))
        print(
            f"\ntraining complete after {self.train_iter} iterations; results saved to '{self.plot_dir}'. "
            "Optimization is frozen at the identified values — close the viewer window to exit.\n"
        )

    def _parameter_sets(self):
        """Return the (nominal CAD, current identified) parameter dictionaries."""
        nominal = self._parameter_set(
            np.full(self.dof_count, self.init_ke),
            np.full(self.dof_count, self.init_kd),
            np.zeros(self.dof_count),
            np.zeros(self.dof_count),
            np.ones(self.model.body_count),
            np.ones(self.model.body_count),
        )
        identified = self._parameter_set(
            self.ke_param.numpy(),
            self.kd_param.numpy(),
            self.damping_param.numpy(),
            self.friction_param.numpy(),
            self.mass_scale.numpy(),
            self.inertia_scale.numpy(),
        )
        return nominal, identified

    def save_trajectory_plot(self, suffix=""):
        """Open-loop replay the recording with the nominal and current identified
        parameters and save a measured-vs-simulated trajectory plot.

        Args:
            suffix: Appended to the base file name (e.g. the iteration count).
        """
        nominal, identified = self._parameter_sets()
        # open-loop replay of the whole recording with each parameter set, driven
        # by the recorded command and started from the measured pose
        nominal_traj = self._open_loop_traj(**nominal)
        identified_traj = self._open_loop_traj(**identified)  # leaves model at identified
        path = os.path.join(self.plot_dir, f"{PLOT_BASENAME}{suffix}.png")
        self._plot(self._pos_np, nominal_traj, identified_traj, path)
        return identified_traj

    def _save_parameters_csv(self, nominal, identified, path):
        """Write the nominal and identified parameters to a CSV file."""
        arm_names = list(CSV_JOINT_TO_USD.values())
        body_names = [label.rsplit("/", 1)[-1] for label in self.model.body_label]
        mass_ref = self.body_mass_ref.numpy()
        rows = [("category", "name", "parameter", "unit", "nominal", "identified")]
        joint_params = (
            ("target_ke", "N*m/rad", "ke"),
            ("target_kd", "N*m*s/rad", "kd"),
            ("damping", "N*m*s/rad", "damping"),
            ("friction", "N*m", "friction"),
        )
        for k, dof in enumerate(self.arm_dof_list):
            for label, unit, key in joint_params:
                rows.append(("joint", arm_names[k], label, unit, nominal[key][dof], identified[key][dof]))
        for b, bname in enumerate(body_names):
            rows.append(
                (
                    "link",
                    bname,
                    "mass",
                    "kg",
                    mass_ref[b] * nominal["mass_scale"][b],
                    mass_ref[b] * identified["mass_scale"][b],
                )
            )
            rows.append(
                ("link", bname, "inertia_scale", "-", nominal["inertia_scale"][b], identified["inertia_scale"][b])
            )
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow([f"{v:.6g}" if isinstance(v, (float, np.floating)) else v for v in row])
        print(f"identified parameters saved to {path}")

    def _save_animation(self, identified_traj, path):
        """Save a GIF overlaying two arms: one driven by the identified-parameter
        simulation (solid) and one replaying the measured ground-truth data
        (gray shadow), so their difference is visible.
        """
        if not hasattr(self.viewer, "get_frame"):
            print("animation needs a frame-capturing viewer (--viewer gl); skipping.")
            return
        try:
            from PIL import Image  # noqa: PLC0415
        except ImportError:
            print("PIL not available; skipping animation.")
            return

        # build a 2-world visualization model: world 0 = identified sim,
        # world 1 = measured ground truth, overlaid at the same location
        arm = newton.ModelBuilder()
        arm.add_usd(
            newton.examples.get_asset("so101.usd"),
            enable_self_collisions=False,
            collapse_fixed_joints=False,
            hide_collision_shapes=True,
        )
        viz_builder = newton.ModelBuilder()
        viz_builder.replicate(arm, 2, spacing=(0.0, 0.0, 0.0))
        viz_builder.add_ground_plane()
        viz_model = viz_builder.finalize()

        # tint the ground-truth arm (world 1) as a gray shadow
        colors = viz_model.shape_color.numpy()
        worlds = viz_model.shape_world.numpy()
        colors[worlds == 1] = (0.35, 0.4, 0.5)
        viz_model.shape_color.assign(colors)

        self.viewer.set_model(viz_model)
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))  # overlay the two arms
        self.viewer.set_camera(pos=wp.vec3(0.55, 0.55, 0.4), pitch=-18.0, yaw=-135.0)

        state = viz_model.state()
        dof = self.dof_count
        # animate the fitted (model-valid) span; past it the open-loop sim
        # resonates beyond the model's bandwidth
        n_end = min(self.fit_boundary, self.num_frames)
        stride = max(n_end // 150, 1)
        q = state.joint_q.numpy()
        frames = []
        for f in range(0, n_end, stride):
            q[0:dof] = identified_traj[f]  # world 0: identified-parameter sim
            q[dof : 2 * dof] = self._pos_np[f]  # world 1: measured ground truth
            state.joint_q.assign(q)
            newton.eval_fk(viz_model, state.joint_q, state.joint_qd, state)
            self.viewer.begin_frame(f * self.frame_dt)
            self.viewer.log_state(state)
            self.viewer.end_frame()
            # get_frame() already returns a top-left-origin image (the GL
            # bottom-left framebuffer is flipped inside the readback kernel),
            img = self.viewer.get_frame().numpy()
            im = Image.fromarray(img)
            im = im.resize((480, max(1, 480 * im.height // im.width)), Image.BILINEAR)
            frames.append(im)

        # restore the training model so the interactive loop keeps working
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.55, 0.55, 0.4), pitch=-18.0, yaw=-135.0)

        if frames:
            frames[0].save(
                path,
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 * self.frame_dt * stride),
                loop=0,
            )
            print(f"validation animation saved to {path} (solid: identified, shadow: ground truth)")

    @staticmethod
    def _parameter_set(ke, kd, damping, friction, mass_scale, inertia_scale):
        return {
            "ke": np.asarray(ke, np.float32),
            "kd": np.asarray(kd, np.float32),
            "damping": np.asarray(damping, np.float32),
            "friction": np.asarray(friction, np.float32),
            "mass_scale": np.asarray(mass_scale, np.float32),
            "inertia_scale": np.asarray(inertia_scale, np.float32),
        }

    def _print_comparison(self, nominal, identified):
        arm = self.arm_dof_list
        arm_names = list(CSV_JOINT_TO_USD.values())
        body_names = [label.rsplit("/", 1)[-1] for label in self.model.body_label]
        mass_ref = self.body_mass_ref.numpy()
        rmse = np.rad2deg(np.sqrt(self.loss_history[-1])) if self.loss_history else float("nan")
        print("\n" + "=" * 60)
        print("SO-101 system identification on real chirp data")
        print(f"fitted {self.num_slots} windows over {self.train_iter} steps, window rmse {rmse:.2f} deg")
        print("real hardware has no ground-truth parameters; 'nominal' is the")
        print("CAD / initial model the identified values are compared against.")
        print("-" * 60)
        print(f"{'joint':<13}{'parameter':<11}{'nominal':>11}{'identified':>13}")
        for k, dof in enumerate(arm):
            for label, key in (
                ("target_ke", "ke"),
                ("target_kd", "kd"),
                ("damping", "damping"),
                ("friction", "friction"),
            ):
                jname = arm_names[k] if label == "target_ke" else ""
                print(f"{jname:<13}{label:<11}{nominal[key][dof]:>11.4f}{identified[key][dof]:>13.4f}")
        print("-" * 60)
        print(f"{'link':<13}{'parameter':<11}{'nominal':>11}{'identified':>13}")
        for b, bname in enumerate(body_names):
            print(
                f"{bname:<13}{'mass [kg]':<11}{mass_ref[b] * nominal['mass_scale'][b]:>11.4f}{mass_ref[b] * identified['mass_scale'][b]:>13.4f}"
            )
            print(f"{'':<13}{'I-scale':<11}{nominal['inertia_scale'][b]:>11.4f}{identified['inertia_scale'][b]:>13.4f}")
        print("=" * 60 + "\n")

    def _open_loop_traj(self, ke, kd, damping, friction, mass_scale, inertia_scale):
        """Free-running forward rollout of the whole recording with a parameter set.

        Driven by the recorded command (zero-order hold) and started from the
        first measured pose. Returns simulated joint angles, shape
        ``[num_frames, dof_count]`` [rad].
        """
        model = self.model
        model.joint_target_ke.assign(ke)
        model.joint_target_kd.assign(kd)
        model.joint_damping.assign(damping)
        friction_arr = wp.array(friction, dtype=float)
        mass_scale_arr = wp.array(mass_scale, dtype=float)
        inertia_scale_arr = wp.array(inertia_scale, dtype=float)
        wp.launch(
            compute_spatial_inertia_kernel,
            dim=model.body_count,
            inputs=[self.body_mass_ref, self.body_inertia_ref, mass_scale_arr, inertia_scale_arr],
            outputs=[self.solver.body_I_m],
        )

        s0, s1 = model.state(), model.state()
        control = model.control()
        s0.joint_q.assign(self._pos_np[0])
        s0.joint_qd.assign(self._vel_np[0])
        newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)

        traj = np.empty((self.num_frames, self.dof_count), dtype=np.float32)
        for f in range(self.num_frames):
            control.joint_target_q.assign(self._cmd_np[f])
            for _ in range(self.sim_substeps):
                wp.launch(
                    friction_torque_kernel,
                    dim=self.dof_count,
                    inputs=[s0.joint_qd, friction_arr, self.friction_vel_eps],
                    outputs=[control.joint_f],
                )
                self.solver.step(s0, s1, control, None, self.sim_dt)
                s0, s1 = s1, s0
            traj[f] = s0.joint_q.numpy()
        return traj

    def _plot_loss_curve(self, path):
        """Plot the per-iteration trajectory loss over the whole training run."""
        try:
            import matplotlib.pyplot as plt  # noqa: PLC0415
        except ImportError:
            print("matplotlib not available; skipping loss curve plot.")
            return
        if not self.loss_history:
            return

        loss = np.array(self.loss_history)
        iters = np.arange(len(loss))
        rmse_deg = np.rad2deg(np.sqrt(loss))

        fig, ax_loss = plt.subplots(figsize=(8, 4.5))
        ax_loss.plot(iters, loss, color="tab:blue", lw=1.4)
        ax_loss.set_yscale("log")
        ax_loss.set_xlabel("optimizer step")
        ax_loss.set_ylabel("trajectory loss [rad$^2$]", color="tab:blue")
        ax_loss.tick_params(axis="y", labelcolor="tab:blue")
        ax_loss.grid(True, which="both", alpha=0.3)

        # secondary axis in the more interpretable window RMSE [deg]
        ax_rmse = ax_loss.twinx()
        ax_rmse.plot(iters, rmse_deg, color="tab:orange", lw=1.0, alpha=0.0)
        ax_rmse.set_ylabel("window rmse [deg]", color="tab:orange")
        ax_rmse.set_yscale("log")
        ax_rmse.set_ylim(np.rad2deg(np.sqrt(ax_loss.get_ylim())))
        ax_rmse.tick_params(axis="y", labelcolor="tab:orange")

        ax_loss.set_title(f"SO-101 system identification: training loss ({len(loss)} steps)")
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"loss curve saved to {path}")

    def _plot(self, measured, nominal_traj, identified_traj, path):
        try:
            import matplotlib.pyplot as plt  # noqa: PLC0415
        except ImportError:
            print("matplotlib not available; skipping trajectory plot.")
            return

        arm = self.arm_dof_list
        arm_names = list(CSV_JOINT_TO_USD.values())
        t = np.arange(self.num_frames) * self.frame_dt
        t_fit = min(self.fit_boundary, self.num_frames - 1) * self.frame_dt
        rmse = np.rad2deg(np.sqrt(self.loss_history[-1])) if self.loss_history else float("nan")

        fig, axes = plt.subplots(len(arm), 1, figsize=(10, 2.0 * len(arm)), sharex=True)
        for ax, dof, name in zip(axes, arm, arm_names, strict=True):
            meas = np.rad2deg(measured[:, dof])
            ax.plot(t, meas, color="black", lw=1.6, label="measured (ground truth)")
            ax.plot(t, np.rad2deg(identified_traj[:, dof]), color="tab:orange", lw=1.2, label="identified params")
            ax.plot(t, np.rad2deg(nominal_traj[:, dof]), color="0.6", lw=1.0, ls="--", label="nominal CAD params")
            ax.axvline(t_fit, color="tab:blue", lw=0.8, ls=":")
            ax.set_ylabel(f"{name}\n[deg]")
            ax.grid(True, alpha=0.3)
            # clip to the measured range: past the fitted region the open-loop sim
            # resonates beyond the model's drive bandwidth and would dominate the axis
            pad = 0.4 * (meas.max() - meas.min()) + 5.0
            ax.set_ylim(meas.min() - pad, meas.max() + pad)
        axes[0].legend(loc="upper right", fontsize=8, ncol=3)
        axes[0].set_title(
            f"SO-101 system identification: simulated vs measured joint trajectories "
            f"(iter {self.train_iter}, window rmse {rmse:.2f} deg)"
        )
        axes[-1].set_xlabel("time [s]  (dotted line: end of fitted / model-valid region)")
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"trajectory plot saved to {path}")

    def forward_kinematics_ee(self, q_rad):
        """End-effector positions along a joint-space trajectory, shape (T, 3)."""
        scratch = self.model.state()
        qd = wp.zeros_like(self.model.joint_qd)
        traj = np.empty((self.num_frames, 3))
        q = self.model.joint_q.numpy()
        for f in range(self.num_frames):
            q[: self.dof_count] = q_rad[f]
            self.model.joint_q.assign(q)
            newton.eval_fk(self.model, self.model.joint_q, qd, scratch)
            traj[f] = scratch.body_q.numpy()[self.ee_body, :3]
        return traj

    def render(self):
        if self.viewer.is_paused():
            self.viewer.begin_frame(self.viewer.time)
            self.viewer.end_frame()
            return

        # after training stops, gently loop the last window one frame per call so
        # the viewer keeps pacing/processing events (close the window to exit)
        if self._finalized:
            states = self.slot_states[0]
            i = self._idle_frame % self.window_len
            self._idle_frame += 1
            self.viewer.begin_frame(i * self.frame_dt)
            self.viewer.log_state(states[(i + 1) * self.sim_substeps])
            self.viewer.end_frame()
            return

        # replay the most recent minibatch's first window every few iterations
        if self.train_iter > 0 and (self.train_iter - 1) % self.render_interval != 0:
            return

        f0 = int(self.starts.numpy()[0])
        states = self.slot_states[0]
        sim_ee = np.array(
            [states[(i + 1) * self.sim_substeps].body_q.numpy()[self.ee_body, :3] for i in range(self.window_len)]
        )
        # measured end-effector path (static, full trajectory) and the simulated
        # path over the currently displayed window
        ref_starts = wp.array(self.ee_traj_ref[:-1], dtype=wp.vec3)
        ref_ends = wp.array(self.ee_traj_ref[1:], dtype=wp.vec3)
        ref_colors = wp.full(self.num_frames - 1, wp.vec3(0.2, 0.8, 0.3), dtype=wp.vec3)
        sim_starts = wp.array(sim_ee[:-1], dtype=wp.vec3)
        sim_ends = wp.array(sim_ee[1:], dtype=wp.vec3)
        sim_colors = wp.full(self.window_len - 1, wp.vec3(1.0, 0.5, 0.1), dtype=wp.vec3)

        for i in range(self.window_len):
            self.viewer.begin_frame((f0 + i) * self.frame_dt)
            if self.loss_history:
                self.viewer.log_scalar("/loss", self.loss_history[-1])
            self.viewer.log_state(states[(i + 1) * self.sim_substeps])
            self.viewer.log_lines("/ee_traj_measured", ref_starts, ref_ends, ref_colors)
            self.viewer.log_lines("/ee_traj_sim", sim_starts, sim_ends, sim_colors)
            self.viewer.end_frame()

    def test_post_step(self):
        assert np.isfinite(self.loss_history[-1]), "loss must stay finite during training"

    def test_final(self):
        self.finalize()
        loss_history = np.array(self.loss_history)
        assert len(loss_history) >= 2
        assert np.isfinite(loss_history).all()
        # compare averaged windows of the loss to smooth out minibatch noise
        head = loss_history[: max(len(loss_history) // 4, 1)].mean()
        tail = loss_history[-max(len(loss_history) // 4, 1) :].mean()
        assert tail < head, f"trajectory loss must decrease (head={head:.3e} tail={tail:.3e})"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Path to a recorded chirp CSV.")
        parser.add_argument("--stride", type=int, default=1, help="Decimate the recording by this factor.")
        parser.add_argument("--max-frames", type=int, default=0, help="Cap the number of frames (0 = all).")
        parser.add_argument("--substeps", type=int, default=4, help="Simulation substeps per recorded frame.")
        parser.add_argument("--window", type=int, default=15, help="Frames per truncated-backprop window.")
        parser.add_argument(
            "--num-windows", type=int, default=6, help="Number of evenly-spaced windows fit each optimizer step."
        )
        parser.add_argument(
            "--fit-fraction",
            type=float,
            default=0.6,
            help="Fraction of the chirp (from the low-frequency start) to fit; keeps windows within the model's drive bandwidth.",
        )
        parser.add_argument(
            "--train-iters",
            type=int,
            default=80,
            help="Optimizer steps after which the nominal-vs-identified report and plot are produced.",
        )
        parser.add_argument(
            "--plot",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Save measured-vs-simulated trajectory plots (PNG).",
        )
        parser.add_argument(
            "--animate",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Save a GIF overlaying the identified-parameter arm and the ground-truth shadow (needs --viewer gl).",
        )
        parser.add_argument(
            "--plot-dir",
            type=str,
            default="plot",
            help="Directory the trajectory plots are written to.",
        )
        parser.add_argument(
            "--plot-interval",
            type=int,
            default=10,
            help="Save a trajectory plot every this many optimizer steps (0 to disable periodic plots).",
        )
        parser.add_argument(
            "--verbose", action="store_true", help="Print out additional status messages during execution."
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
