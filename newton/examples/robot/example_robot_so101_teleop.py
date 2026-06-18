# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot SO-101 Teleop
#
# Keyboard teleoperation of the SO-ARM101 end effector. The keyboard moves
# a target pose for the gripper, an IK solver (position + rotation
# objectives) converts it into joint position targets, and the MuJoCo
# solver drives track them dynamically.
#
# Keys (world frame):
#   I / K : move +X / -X        Z / X : rotate about world X
#   J / L : move +Y / -Y        C / V : rotate about world Y
#   U / O : move up / down      B / N : rotate about world Z
#   T / G : open / close jaw    P     : reset target to current pose
#
# The target can also be dragged with the mouse gizmo.
#
# Command: python -m newton.examples robot_so101_teleop
#
###########################################################################

import itertools

import warp as wp

import newton
import newton.examples
import newton.ik as ik

CONTROLS = """
SO-101 keyboard teleop (world frame):
  I / K : move +X / -X        Z / X : rotate about world X
  J / L : move +Y / -Y        C / V : rotate about world Y
  U / O : move up / down      B / N : rotate about world Z
  T / G : open / close jaw    P     : reset target to current pose
"""


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
        self.fps = 60
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
        for joint_idx, label in enumerate(builder.joint_label):
            name = label.rsplit("/", 1)[-1]
            if name in initial_pose:
                builder.joint_q[builder.joint_q_start[joint_idx]] = initial_pose[name]
            if name == "Jaw":
                self.jaw_coord = builder.joint_q_start[joint_idx]

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

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.7, 0.7, 0.45), pitch=-17.0, yaw=-135.0)

        print(CONTROLS)

        self.capture()

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

        # without key input the target stays at the initial pose, so the
        # gripper must hold position
        body_q = self.state_0.body_q.numpy()
        ee_pos = body_q[self.ee_index][:3]
        target_pos = wp.transform_get_translation(self.target_tf)
        error = float(wp.length(wp.vec3(*ee_pos) - target_pos))
        assert error < 0.05, f"end effector must track the target pose, error={error}"


if __name__ == "__main__":
    viewer, args = newton.examples.init()

    newton.examples.run(Example(viewer, args), args)
