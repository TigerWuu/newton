# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot SO-101 Digital Twin
#
# Keyboard teleoperation of a *digital twin* of the SO-ARM101 arm: a Newton
# simulation whose joint drives and inertial properties have been replaced
# with the values identified from the real hardware (see
# ``example_diffsim_so101_sysid_true``), running in lock-step with the
# physical robot.
#
# This combines two pieces:
#
#   1. Teleop control (shared with ``example_robot_so101_teleop``): the
#      keyboard moves a target pose for the gripper, an IK solver converts it
#      into joint position targets, and the MuJoCo solver tracks them.
#
#   2. A real-hardware bridge. The same joint targets that drive the
#      simulation are streamed to the physical SO-101 at 50 Hz through the
#      lerobot ``SO101Follower`` driver -- the very driver used by
#      ``record_tra.sh`` to capture the system-identification trajectories.
#      The simulation therefore acts as a digital twin: it is parameterized
#      from the real arm and commands it in real time.
#
# The model is loaded from the SO-101 USD asset and then overwritten with the
# parameters in ``identified_parameters.csv`` (per-joint servo stiffness /
# damping, viscous damping and Coulomb friction, and per-link mass / inertia
# scale), so the twin reproduces the measured dynamics rather than the nominal
# CAD ones.
#
# Keys (world frame):
#   I / K : move +X / -X        Z / X : rotate about world X
#   J / L : move +Y / -Y        C / V : rotate about world Y
#   U / O : move up / down      B / N : rotate about world Z
#   T / G : open / close jaw    P     : reset target to current pose
#
# Commands:
#   # simulation only (no hardware needed), auto-loads the latest identified CSV
#   python -m newton.examples robot_so101_digital_twin --no-hardware
#   # drive the physical arm too
#   python -m newton.examples robot_so101_digital_twin \
#       --robot-port /dev/followerarm-right --robot-id my_awesome_follower_arm
#
###########################################################################

import csv
import glob
import itertools
import math
import os
import time

import warp as wp

import newton
import newton.examples
import newton.ik as ik

CONTROLS = """
SO-101 digital-twin keyboard teleop (world frame):
  I / K : move +X / -X        Z / X : rotate about world X
  J / L : move +Y / -Y        C / V : rotate about world Y
  U / O : move up / down      B / N : rotate about world Z
  T / G : open / close jaw    P     : reset target to current pose
"""

# Map the USD joint name to the lerobot motor name driven on the real arm.
# This is the inverse of the recorder's channel map (see record_tra.sh and
# example_diffsim_so101_sysid_true.CSV_JOINT_TO_USD); the gripper (Jaw) is
# commanded separately as a 0-100 percentage.
USD_JOINT_TO_MOTOR = {
    "Rotation": "shoulder_pan",
    "Pitch": "shoulder_lift",
    "Elbow": "elbow_flex",
    "Wrist_Pitch": "wrist_flex",
    "Wrist_Roll": "wrist_roll",
}
GRIPPER_MOTOR = "gripper"

# Real-robot <-> USD joint-frame calibration, mirroring the identification
# script: degrees with 0 deg at each joint's calibrated mid-range, unit sign and
# no offset. Flip a sign or add an offset [rad] here if a channel is mirrored or
# shifted on your hardware.
JOINT_SIGN = dict.fromkeys(USD_JOINT_TO_MOTOR, +1.0)
JOINT_OFFSET_RAD = dict.fromkeys(USD_JOINT_TO_MOTOR, 0.0)


def find_latest_identified_csv(plot_dir="plot"):
    """Return the most recent ``identified_parameters.csv`` under ``plot_dir``.

    The system-identification example writes one CSV per run into a
    timestamped sub-directory; the newest is the natural default for the twin.
    Returns None if none exist.
    """
    matches = glob.glob(os.path.join(plot_dir, "*", "identified_parameters.csv"))
    matches += glob.glob(os.path.join(plot_dir, "identified_parameters.csv"))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def load_identified_parameters(path):
    """Parse an ``identified_parameters.csv`` produced by the sysid example.

    Returns:
        joint: dict keyed by USD joint name -> {"target_ke", "target_kd",
            "damping", "friction"} [SI].
        link: dict keyed by link name -> {"mass" [kg], "inertia_scale" [-]}.
    """
    joint: dict[str, dict[str, float]] = {}
    link: dict[str, dict[str, float]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            value = float(row["identified"])
            if row["category"] == "joint":
                joint.setdefault(row["name"], {})[row["parameter"]] = value
            elif row["category"] == "link":
                link.setdefault(row["name"], {})[row["parameter"]] = value
    return joint, link


def apply_identified_parameters(builder, joint_params, link_params, armature):
    """Overwrite the freshly-imported SO-101 model with identified values.

    Joint drive/friction parameters are written per DoF (every SO-101 joint is
    1-DoF revolute, so the coordinate index equals the DoF index); link mass and
    inertia are replaced/scaled from the nominal CAD values. Mutating the builder
    lists before ``finalize()`` keeps the dependent inverse-mass / inverse-inertia
    arrays consistent.
    """
    for joint_idx, label in enumerate(builder.joint_label):
        name = label.rsplit("/", 1)[-1]
        dof = builder.joint_q_start[joint_idx]
        p = joint_params.get(name)
        if p is not None:
            builder.joint_target_ke[dof] = p["target_ke"]
            builder.joint_target_kd[dof] = p["target_kd"]
            builder.joint_damping[dof] = p["damping"]
            builder.joint_friction[dof] = p["friction"]
        # match the reflected rotor inertia used during identification so the
        # twin reproduces the same stiff-servo dynamics
        builder.joint_armature[dof] = armature

    for body_idx, label in enumerate(builder.body_label):
        name = label.rsplit("/", 1)[-1]
        p = link_params.get(name)
        if p is None:
            continue
        mass = p["mass"]
        builder.body_mass[body_idx] = mass
        builder.body_inv_mass[body_idx] = 1.0 / mass if mass > 0.0 else 0.0
        inertia = builder.body_inertia[body_idx] * p["inertia_scale"]
        builder.body_inertia[body_idx] = inertia
        builder.body_inv_inertia[body_idx] = wp.inverse(inertia) if mass > 0.0 else inertia


class SO101Hardware:
    """Streams simulated joint targets to a physical SO-101 follower arm.

    Wraps the same lerobot ``SO101Follower`` driver used by ``record_tra.sh``.
    Arm joints are commanded in degrees (0 deg = calibrated mid-range); the jaw
    is commanded as a 0-100 opening percentage. lerobot is imported lazily so the
    example still runs (simulation only) on machines without it installed.
    """

    def __init__(self, port, robot_id, fps):
        self.port = port
        self.robot_id = robot_id
        self.fps = fps
        self.robot = None

    def connect(self):
        # imported here so the example has no hard dependency on lerobot
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig  # noqa: PLC0415
        from lerobot.robots.so_follower.so_follower import SO101Follower  # noqa: PLC0415

        cfg = SOFollowerRobotConfig(port=self.port, id=self.robot_id, use_degrees=True)
        self.robot = SO101Follower(cfg)
        self.robot.connect()

    def read_arm_deg(self, motors):
        """Current measured positions [deg] for the given motor names."""
        obs = self.robot.get_observation()
        return {m: float(obs[f"{m}.pos"]) for m in motors}

    def move_to(self, command, secs):
        """Cosine-eased move from the current pose to ``command`` over ``secs``.

        Avoids a jump when the real arm is first slaved to the simulated pose.
        ``command`` maps ``"{motor}.pos"`` -> value (deg for arm, 0-100 jaw).
        """
        motors = [k.rsplit(".", 1)[0] for k in command]
        start = self.read_arm_deg(motors)
        n = max(1, int(round(secs * self.fps)))
        t0 = time.perf_counter()
        for i in range(1, n + 1):
            a = 0.5 * (1.0 - math.cos(math.pi * i / n))  # 0 -> 1, ease in/out
            cmd = {f"{m}.pos": start[m] + (command[f"{m}.pos"] - start[m]) * a for m in motors}
            self.robot.send_action(cmd)
            self._sleep_until(t0 + i / self.fps)

    def send(self, command):
        """Send one joint-target frame (non-blocking)."""
        self.robot.send_action(command)

    def disconnect(self):
        if self.robot is not None:
            self.robot.disconnect()  # releases torque
            self.robot = None

    @staticmethod
    def _sleep_until(deadline):
        dt = deadline - time.perf_counter()
        if dt > 0:
            time.sleep(dt)


@wp.kernel
def assign_joint_targets_kernel(
    ik_joint_q: wp.array2d[wp.float32],
    jaw_target: wp.array[wp.float32],
    jaw_coord: wp.int32,
    limit_lower: wp.array[wp.float32],
    limit_upper: wp.array[wp.float32],
    # output
    joint_target_q: wp.array[wp.float32],
):
    # all SO-101 joints are 1-dof revolute, so coord index == dof index
    i = wp.tid()
    target = ik_joint_q[0, i]
    if i == jaw_coord:
        target = jaw_target[0]
    joint_target_q[i] = wp.clamp(target, limit_lower[i], limit_upper[i])


class Example:
    def __init__(self, viewer, args):
        # match the 50 Hz control rate the real arm is commanded at (and that the
        # identification trajectories were recorded at)
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.device = wp.get_device()

        # teleop speeds
        self.linear_speed = 0.2  # m/s
        self.angular_speed = 1.2  # rad/s
        self.jaw_speed = 1.5  # rad/s

        # --- load identified parameters ----------------------------------
        params_path = args.params or find_latest_identified_csv(args.plot_dir)
        if params_path is None or not os.path.exists(params_path):
            raise FileNotFoundError(
                "no identified_parameters.csv found. Run "
                "`python -m newton.examples diffsim_so101_sysid_true` first, or pass "
                "--params <path/to/identified_parameters.csv>."
            )
        self.joint_params, self.link_params = load_identified_parameters(params_path)
        print(f"loaded identified parameters from {params_path}")

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        # the USD already contains joint drives (stiffness, damping, effort
        # limits), which add_usd() imports as position-control targets
        builder.add_usd(
            newton.examples.get_asset("so101.usd"),
            enable_self_collisions=False,
            collapse_fixed_joints=False,
            hide_collision_shapes=True,
        )

        # replace the nominal CAD drives/inertia with the identified values so
        # the simulation behaves like the physical arm it is twinning
        apply_identified_parameters(builder, self.joint_params, self.link_params, args.armature)

        # rest pose of the arm ("ready" configuration)
        initial_pose = {
            "Rotation": -0.2736,
            "Pitch": -0.6109,
            "Elbow": -0.0745,
            "Wrist_Pitch": 1.5148,
            "Wrist_Roll": -1.6034,
            "Jaw": -0.1465,
        }
        self.jaw_coord = -1
        # dof index of each motor streamed to the hardware, in canonical order
        self.motor_dof = {}
        for joint_idx, label in enumerate(builder.joint_label):
            name = label.rsplit("/", 1)[-1]
            coord = builder.joint_q_start[joint_idx]
            if name in initial_pose:
                builder.joint_q[coord] = initial_pose[name]
            if name == "Jaw":
                self.jaw_coord = coord
            elif name in USD_JOINT_TO_MOTOR:
                self.motor_dof[name] = coord

        # the gripper body is the end effector controlled by IK
        self.ee_index = next(i for i, label in enumerate(builder.body_label) if label.endswith("/gripper"))

        builder.add_ground_plane()

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverMuJoCo(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # keyboard-driven target pose, initialized to the current gripper pose
        body_q = self.state_0.body_q.numpy()
        self.target_tf = wp.transform(*body_q[self.ee_index])

        # soft reach limit: keep the target inside a sphere around the
        # shoulder pivot so the arm can always track it
        shoulder_index = next(i for i, label in enumerate(builder.body_label) if label.endswith("/shoulder"))
        chain = [wp.vec3(*body_q[i][:3]) for i in range(shoulder_index, self.ee_index + 1)]
        self.reach_center = chain[0]
        self.max_reach = 0.95 * sum(wp.length(b - a) for a, b in itertools.pairwise(chain))

        limit_lower = self.model.joint_limit_lower.numpy()
        limit_upper = self.model.joint_limit_upper.numpy()
        self.jaw_limits = (float(limit_lower[self.jaw_coord]), float(limit_upper[self.jaw_coord]))
        self.jaw_q = float(self.model.joint_q.numpy()[self.jaw_coord])
        self.jaw_target = wp.array([self.jaw_q], dtype=wp.float32, device=self.device)

        # IK setup: solve for joint coordinates that realize the target pose
        rot = wp.transform_get_rotation(self.target_tf)
        # the arm has only 5 dof, so a 6-dof pose target is generically
        # unreachable; weight position far above rotation so the gripper
        # tracks position exactly and orientation best-effort
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.transform_get_translation(self.target_tf)], dtype=wp.vec3),
            weight=50.0,
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.vec4(rot[0], rot[1], rot[2], rot[3])], dtype=wp.vec4),
            weight=1.0,
        )
        self.limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0,
        )

        # warm-started IK variables, kept separate from the simulated state
        self.ik_joint_q = wp.clone(self.model.joint_q).reshape((1, self.model.joint_coord_count))
        self.ik_iters = 24
        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.limit_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        self._reset_key_prev = False

        # --- real-hardware bridge ----------------------------------------
        # never touch hardware in the automated test, even if a robot is plugged
        # in: the test runs headless with no operator at the power switch
        self.hardware = None
        if args.hardware and not args.test:
            self.hardware = SO101Hardware(args.robot_port, args.robot_id, self.fps)
            try:
                self.hardware.connect()
                print(f"connected to SO-101 on {args.robot_port} (id={args.robot_id})")
                # ease the real arm into the simulated rest pose before slaving it
                self.hardware.move_to(self._hardware_command(), args.init_move_secs)
            except Exception as exc:
                print(f"hardware unavailable ({exc!r}); running simulation only.")
                self.hardware = None
        self._next_send = time.perf_counter()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.7, 0.7, 0.45), pitch=-17.0, yaw=-135.0)

        print(CONTROLS)

        self.capture()

    def _hardware_command(self):
        """Build the ``{motor}.pos`` command dict from the current joint targets.

        Arm joints are converted from the USD frame [rad] to the hardware frame
        [deg]; the jaw maps linearly to a 0-100 opening percentage.
        """
        target_q = self.control.joint_target_q.numpy()
        command = {}
        for name, dof in self.motor_dof.items():
            sign, offset = JOINT_SIGN[name], JOINT_OFFSET_RAD[name]
            deg = math.degrees(sign * (float(target_q[dof]) - offset))
            command[f"{USD_JOINT_TO_MOTOR[name]}.pos"] = deg
        lo, hi = self.jaw_limits
        pct = 100.0 * (self.jaw_q - lo) / (hi - lo) if hi > lo else 0.0
        command[f"{GRIPPER_MOTOR}.pos"] = min(100.0, max(0.0, pct))
        return command

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)

        wp.launch(
            assign_joint_targets_kernel,
            dim=self.model.joint_coord_count,
            inputs=[
                self.ik_joint_q,
                self.jaw_target,
                self.jaw_coord,
                self.model.joint_limit_lower,
                self.model.joint_limit_upper,
            ],
            outputs=[self.control.joint_target_q],
            device=self.device,
        )

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _send_to_hardware(self):
        """Stream the latest joint targets to the real arm, paced at ``fps``."""
        if self.hardware is None:
            return
        self.hardware._sleep_until(self._next_send)
        self._next_send = time.perf_counter() + self.frame_dt
        self.hardware.send(self._hardware_command())

    def _key_axis(self, pos_key: str, neg_key: str) -> float:
        return float(self.viewer.is_key_down(pos_key)) - float(self.viewer.is_key_down(neg_key))

    def _reset_target(self):
        """Snap the target back to the current simulated gripper pose."""
        body_q = self.state_0.body_q.numpy()
        self.target_tf = wp.transform(*body_q[self.ee_index])
        self.jaw_q = float(self.state_0.joint_q.numpy()[self.jaw_coord])
        wp.copy(self.ik_joint_q, self.state_0.joint_q.reshape((1, self.model.joint_coord_count)))

    def _apply_keyboard(self):
        if not hasattr(self.viewer, "is_key_down"):
            return

        reset_down = bool(self.viewer.is_key_down("p"))
        if reset_down and not self._reset_key_prev:
            self._reset_target()
        self._reset_key_prev = reset_down

        pos = wp.transform_get_translation(self.target_tf)
        rot = wp.transform_get_rotation(self.target_tf)

        move = wp.vec3(self._key_axis("i", "k"), self._key_axis("j", "l"), self._key_axis("u", "o"))
        if wp.length_sq(move) > 0.0:
            pos = pos + wp.normalize(move) * (self.linear_speed * self.frame_dt)

        for pos_key, neg_key, axis in (
            ("z", "x", wp.vec3(1.0, 0.0, 0.0)),
            ("c", "v", wp.vec3(0.0, 1.0, 0.0)),
            ("b", "n", wp.vec3(0.0, 0.0, 1.0)),
        ):
            angle = self._key_axis(pos_key, neg_key) * self.angular_speed * self.frame_dt
            if angle != 0.0:
                rot = wp.normalize(wp.quat_from_axis_angle(axis, angle) * rot)

        self.target_tf = wp.transform(pos, rot)

        jaw_lower, jaw_upper = self.jaw_limits
        self.jaw_q += self._key_axis("t", "g") * self.jaw_speed * self.frame_dt
        self.jaw_q = min(max(self.jaw_q, jaw_lower), jaw_upper)

    def _push_targets(self):
        """Clamp the keyboard/gizmo-updated target and push it into the IK objectives."""
        pos = wp.transform_get_translation(self.target_tf)
        pos = wp.vec3(pos[0], pos[1], max(pos[2], 0.02))
        offset = pos - self.reach_center
        dist = wp.length(offset)
        if dist > self.max_reach:
            pos = self.reach_center + offset * (self.max_reach / dist)
        self.target_tf = wp.transform(pos, wp.transform_get_rotation(self.target_tf))

        self.pos_obj.set_target_position(0, pos)
        rot = wp.transform_get_rotation(self.target_tf)
        self.rot_obj.set_target_rotation(0, wp.vec4(rot[0], rot[1], rot[2], rot[3]))
        self.jaw_target.fill_(self.jaw_q)

    def step(self):
        self._apply_keyboard()
        self._push_targets()

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self._send_to_hardware()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)

        # the gizmo mutates self.target_tf in place while dragged
        if hasattr(self.viewer, "log_gizmo"):
            self.viewer.log_gizmo("ee_target", self.target_tf)

        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        joint_q = self.state_0.joint_q.numpy()
        lower = self.model.joint_limit_lower.numpy()
        upper = self.model.joint_limit_upper.numpy()
        limit_eps = 1e-3
        assert ((joint_q >= lower - limit_eps) & (joint_q <= upper + limit_eps)).all(), (
            "joint positions must stay within limits"
        )

        # the identified parameters must have been written into the model
        ke = self.model.joint_target_ke.numpy()
        for name, dof in self.motor_dof.items():
            assert math.isclose(ke[dof], self.joint_params[name]["target_ke"], rel_tol=1e-4), (
                f"identified target_ke not applied to {name}"
            )

        # without key input the target stays at the initial pose, so the
        # gripper must hold position (softer servo gains than the CAD model -> a
        # looser bound than the nominal teleop)
        body_q = self.state_0.body_q.numpy()
        ee_pos = body_q[self.ee_index][:3]
        target_pos = wp.transform_get_translation(self.target_tf)
        error = float(wp.length(wp.vec3(*ee_pos) - target_pos))
        assert error < 0.1, f"end effector must track the target pose, error={error}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--params",
            type=str,
            default=None,
            help="Path to an identified_parameters.csv (default: latest under --plot-dir).",
        )
        parser.add_argument(
            "--plot-dir",
            type=str,
            default="plot",
            help="Directory searched for the latest identified_parameters.csv.",
        )
        parser.add_argument(
            "--armature",
            type=float,
            default=0.1,
            help="Reflected servo rotor inertia, matching the identification setup.",
        )
        parser.add_argument(
            "--hardware",
            action="store_true",
            default=False,
            help="Stream joint targets to a physical SO-101 via lerobot.",
        )
        parser.add_argument(
            "--no-hardware",
            dest="hardware",
            action="store_false",
            help="Run the simulation only (default).",
        )
        parser.add_argument(
            "--robot-port",
            type=str,
            default=os.environ.get("ROBOT_PORT", "/dev/followerarm-right"),
            help="Serial port of the SO-101 follower arm.",
        )
        parser.add_argument(
            "--robot-id",
            type=str,
            default=os.environ.get("ROBOT_ID", "my_awesome_follower_arm"),
            help="lerobot calibration id of the SO-101 follower arm.",
        )
        parser.add_argument(
            "--init-move-secs",
            type=float,
            default=4.0,
            help="Seconds to ease the real arm into the simulated rest pose on connect.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
