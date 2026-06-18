# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot SO-101
#
# Shows how to set up a simulation of the SO-ARM101 low-cost robot arm
# from a USD file using newton.ModelBuilder.add_usd() and drives the
# joints with smooth sinusoidal position targets around an initial pose.
#
# Command: python -m newton.examples robot_so101
#
###########################################################################

import warp as wp

import newton
import newton.examples


@wp.kernel
def update_joint_targets_kernel(
    time: wp.array[wp.float32],
    dt: wp.float32,
    q_init: wp.array[wp.float32],
    limit_lower: wp.array[wp.float32],
    limit_upper: wp.array[wp.float32],
    dofs_per_world: wp.int32,
    # output
    joint_target_q: wp.array[wp.float32],
):
    world = wp.tid()
    t = time[world] + dt
    time[world] = t
    for i in range(dofs_per_world):
        dof = world * dofs_per_world + i
        # oscillate around the initial pose, staying within the joint limits
        amplitude = 0.25 * (limit_upper[dof] - limit_lower[dof])
        target = q_init[dof] + amplitude * wp.sin(t + 0.5 * float(i + world))
        joint_target_q[dof] = wp.clamp(target, limit_lower[dof], limit_upper[dof])


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = args.world_count

        self.viewer = viewer
        self.device = wp.get_device()

        so101 = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(so101)

        # the USD already contains joint drives (stiffness, damping, effort
        # limits), which add_usd() imports as position-control targets
        so101.add_usd(
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
        for joint_idx, label in enumerate(so101.joint_label):
            name = label.rsplit("/", 1)[-1]
            if name in initial_pose:
                so101.joint_q[so101.joint_q_start[joint_idx]] = initial_pose[name]

        builder = newton.ModelBuilder()
        builder.replicate(so101, self.world_count, spacing=(0.6, 0.6, 0.0))
        builder.add_ground_plane()

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverMuJoCo(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None

        self.dofs_per_world = self.model.joint_dof_count // self.world_count
        self.joint_q_init = wp.clone(self.model.joint_q)
        self.time_step = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.7, 0.7, 0.45), pitch=-17.0, yaw=-135.0)

        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            wp.launch(
                update_joint_targets_kernel,
                dim=self.world_count,
                inputs=[
                    self.time_step,
                    self.sim_dt,
                    self.joint_q_init,
                    self.model.joint_limit_lower,
                    self.model.joint_limit_upper,
                    self.dofs_per_world,
                ],
                outputs=[self.control.joint_target_q],
                device=self.device,
            )

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        joint_q = self.state_0.joint_q.numpy()
        target_q = self.control.joint_target_q.numpy()
        lower = self.model.joint_limit_lower.numpy()
        upper = self.model.joint_limit_upper.numpy()
        limit_eps = 1e-3
        assert ((joint_q >= lower - limit_eps) & (joint_q <= upper + limit_eps)).all(), (
            "joint positions must stay within limits"
        )
        tracking_error = abs(joint_q - target_q).max()
        assert tracking_error < 0.1, f"joints must track position targets, error={tracking_error}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=1)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
