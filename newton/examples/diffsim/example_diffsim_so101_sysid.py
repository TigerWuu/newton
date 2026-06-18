# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Diffsim SO-101 System Identification
#
# Recovers the inertial and joint-level parameters of the SO-ARM101
# low-cost robot arm from recorded joint trajectories by differentiating
# through the simulation with SolverFeatherstone:
#
#   - per-link mass and rotational inertia (as scale factors on the
#     nominal CAD values imported from the USD file)
#   - per-joint viscous damping
#   - per-joint Coulomb (dry) friction
#
# A "real" robot with perturbed ground-truth parameters is simulated to
# record joint encoder data while tracking multi-sine position targets.
# Three worlds with different excitation frequencies run at once to
# enrich the data. The same model, starting from the nominal parameters
# (scale 1, zero damping and friction), is then fitted with Adam to
# match the recorded trajectories.
#
# Viscous damping is differentiable through Model.joint_damping. Dry
# friction is not natively supported by SolverFeatherstone, so it is
# applied as an explicit generalized force -f*tanh(qd/v_eps) through
# Control.joint_f, which keeps the loss differentiable in f. Mass and
# inertia enter through the solver's spatial inertia, which is
# recomputed from the scale parameters inside the tape on every forward
# pass so that gradients reach them.
#
# Damping and friction are recovered almost exactly and the mass scales
# approach their true values, while the inertia scales of the small
# distal links remain only partially identifiable from this excitation
# (their contribution to the measured trajectories is tiny) -- a
# faithful reproduction of what happens when identifying real robots.
#
# Command: python -m newton.examples diffsim_so101_sysid
#
###########################################################################

import numpy as np
import warp as wp
import warp.optim

import newton
import newton.examples


@wp.kernel
def scatter_joint_damping_kernel(
    damping_param: wp.array[float],
    dofs_per_world: int,
    # outputs
    joint_damping: wp.array[float],
):
    world, i = wp.tid()
    joint_damping[world * dofs_per_world + i] = damping_param[i]


@wp.kernel
def compute_spatial_inertia_kernel(
    body_mass_ref: wp.array[float],
    body_inertia_ref: wp.array[wp.mat33],
    mass_scale: wp.array[float],
    inertia_scale: wp.array[float],
    bodies_per_world: int,
    # outputs
    body_I_m: wp.array[wp.spatial_matrix],
):
    world, b = wp.tid()
    body = world * bodies_per_world + b
    m = body_mass_ref[body] * mass_scale[b]
    I = body_inertia_ref[body] * inertia_scale[b]
    # fmt: off
    body_I_m[body] = wp.spatial_matrix(
        m,   0.0, 0.0, 0.0,     0.0,     0.0,
        0.0, m,   0.0, 0.0,     0.0,     0.0,
        0.0, 0.0, m,   0.0,     0.0,     0.0,
        0.0, 0.0, 0.0, I[0, 0], I[0, 1], I[0, 2],
        0.0, 0.0, 0.0, I[1, 0], I[1, 1], I[1, 2],
        0.0, 0.0, 0.0, I[2, 0], I[2, 1], I[2, 2],
    )
    # fmt: on


@wp.kernel
def friction_torque_kernel(
    joint_qd: wp.array[float],
    friction_param: wp.array[float],
    dofs_per_world: int,
    vel_eps: float,
    # outputs
    joint_f: wp.array[float],
):
    world, i = wp.tid()
    dof = world * dofs_per_world + i
    # smooth Coulomb friction so the loss stays differentiable around qd = 0
    joint_f[dof] = -friction_param[i] * wp.tanh(joint_qd[dof] / vel_eps)


@wp.kernel
def trajectory_loss_kernel(
    joint_q: wp.array[float],
    joint_q_ref: wp.array2d[float],
    sample: int,
    norm: float,
    # outputs
    loss: wp.array[float],
):
    tid = wp.tid()
    err = joint_q[tid] - joint_q_ref[sample, tid]
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


class Example:
    def __init__(self, viewer, args):
        self.fps = 30
        self.frame = 0
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 20
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.num_frames = max(int(args.sim_duration * self.fps), 2)
        self.num_worlds = 3

        self.verbose = args.verbose
        self.train_iter = 0
        # rollouts are replayed in the viewer every render_interval iterations
        self.render_interval = 8

        self.viewer = viewer

        # drive gains and armature chosen for stable explicit integration at
        # sim_dt; the much stiffer implicit-friendly drives from the USD would
        # require a far smaller time step with SolverFeatherstone
        drive_ke = 15.0
        drive_kd = 0.05
        armature = 1.0e-4
        self.friction_vel_eps = 0.1

        # rest pose of the arm ("ready" configuration)
        initial_pose = {
            "Rotation": -0.2736,
            "Pitch": -0.6109,
            "Elbow": -0.0745,
            "Wrist_Pitch": 1.5148,
            "Wrist_Roll": -1.6034,
            "Jaw": -0.1465,
        }

        so101 = newton.ModelBuilder()
        so101.add_usd(
            newton.examples.get_asset("so101.usd"),
            enable_self_collisions=False,
            collapse_fixed_joints=False,
            hide_collision_shapes=True,
        )
        for joint_idx, label in enumerate(so101.joint_label):
            name = label.rsplit("/", 1)[-1]
            if name in initial_pose:
                so101.joint_q[so101.joint_q_start[joint_idx]] = initial_pose[name]
        for dof in range(len(so101.joint_target_ke)):
            so101.joint_target_ke[dof] = drive_ke
            so101.joint_target_kd[dof] = drive_kd
            so101.joint_armature[dof] = armature

        builder = newton.ModelBuilder()
        builder.replicate(so101, self.num_worlds, spacing=(0.6, 0.6, 0.0))
        # visual reference only: the arms never touch it and no contacts are computed
        builder.add_ground_plane()

        # use `requires_grad=True` to create a model for differentiable simulation
        self.model = builder.finalize(requires_grad=True)

        self.dof_count = self.model.joint_dof_count
        self.dofs_per_world = self.dof_count // self.num_worlds
        self.bodies_per_world = self.model.body_count // self.num_worlds
        self.ee_body = next(
            i for i, label in enumerate(self.model.body_label[: self.bodies_per_world]) if label.endswith("/gripper")
        )

        self.solver = newton.solvers.SolverFeatherstone(self.model)

        # nominal (CAD) body properties that the scale parameters act on
        self.body_mass_ref = wp.clone(self.model.body_mass, requires_grad=False)
        self.body_inertia_ref = wp.clone(self.model.body_inertia, requires_grad=False)

        # parameters to identify, shared across worlds
        self.damping_param = wp.zeros(self.dofs_per_world, dtype=float, requires_grad=True)
        self.friction_param = wp.zeros(self.dofs_per_world, dtype=float, requires_grad=True)
        self.mass_scale = wp.ones(self.bodies_per_world, dtype=float, requires_grad=True)
        self.inertia_scale = wp.ones(self.bodies_per_world, dtype=float, requires_grad=True)

        # ground-truth parameters of the "real" robot
        rng = np.random.default_rng(42)
        self.mass_scale_true = np.ones(self.bodies_per_world, dtype=np.float32)
        self.inertia_scale_true = np.ones(self.bodies_per_world, dtype=np.float32)
        self.mass_scale_true[1:] = rng.uniform(0.7, 1.3, self.bodies_per_world - 1)
        self.inertia_scale_true[1:] = rng.uniform(0.7, 1.3, self.bodies_per_world - 1)
        self.damping_true = np.array([0.02, 0.02, 0.015, 0.01, 0.005, 0.005], dtype=np.float32)
        self.friction_true = np.array([0.04, 0.04, 0.03, 0.02, 0.01, 0.005], dtype=np.float32)

        # states and controls for the full rollout (one state per substep, as
        # required for the backward pass)
        num_substeps_total = self.num_frames * self.sim_substeps
        self.states = [self.model.state() for _ in range(num_substeps_total + 1)]
        self.controls = [self.model.control() for _ in range(num_substeps_total)]

        # excitation: multi-sine joint position targets around mid-range,
        # smoothly ramped in from the initial pose; each world uses different
        # frequencies and amplitudes for richer, better-conditioned data
        q0 = self.model.joint_q.numpy()
        lower = self.model.joint_limit_lower.numpy()
        upper = self.model.joint_limit_upper.numpy()
        mid = 0.5 * (lower + upper)
        base_freqs = np.array([0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
        freqs = np.empty(self.dof_count)
        amp = np.empty(self.dof_count)
        freq_scales = [1.0, 0.8, 0.4]
        amp_scales = [0.30, 0.25, 0.35]
        for w in range(self.num_worlds):
            sl = slice(w * self.dofs_per_world, (w + 1) * self.dofs_per_world)
            order = base_freqs if w % 2 == 0 else base_freqs[::-1]
            freqs[sl] = order * freq_scales[w % len(freq_scales)]
            amp[sl] = amp_scales[w % len(amp_scales)] * (upper[sl] - lower[sl])
        phase = 0.7 * np.arange(self.dof_count)
        ramp_time = 0.5
        for t in range(num_substeps_total):
            time = t * self.sim_dt
            s = min(time / ramp_time, 1.0)
            s = 0.5 - 0.5 * np.cos(np.pi * s)
            target = mid + amp * np.sin(2.0 * np.pi * freqs * time + phase)
            target = q0 + s * (target - q0)
            self.controls[t].joint_target_q.assign(target.astype(np.float32))

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])

        # recorded joint "encoder" measurements of the ground-truth robot
        self.joint_q_ref = wp.zeros((self.num_frames, self.dof_count), dtype=float, requires_grad=False)
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.loss_history = []

        self.viewer.set_model(self.model)
        # the worlds are already separated by the builder spacing; extra viewer
        # offsets would detach the arms from the trajectory lines drawn below
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))
        self.viewer.set_camera(pos=wp.vec3(1.1, 1.1, 0.6), pitch=-15.0, yaw=-135.0)

        # generate the ground-truth trajectories
        self.damping_param.assign(self.damping_true)
        self.friction_param.assign(self.friction_true)
        self.mass_scale.assign(self.mass_scale_true)
        self.inertia_scale.assign(self.inertia_scale_true)
        self.rollout(record_ref=True)
        self.ee_traj_ref = self.gather_ee_trajectories()

        # reset the parameters to the nominal initial guess
        self.damping_param.zero_()
        self.friction_param.zero_()
        self.mass_scale.fill_(1.0)
        self.inertia_scale.fill_(1.0)

        self.opt_torque = warp.optim.Adam([self.damping_param, self.friction_param], lr=2e-3)
        self.opt_scale = warp.optim.Adam([self.mass_scale, self.inertia_scale], lr=2e-2)

        # capture forward/backward passes
        self.capture()

    def capture(self):
        # warm up once outside the capture: the solver allocates per-state
        # auxiliary arrays on the first step, which must not happen during
        # graph capture
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
        self.rollout(compute_loss=True)
        return self.loss

    def rollout(self, record_ref=False, compute_loss=False):
        # scatter the shared parameters into the per-world model arrays inside
        # the tape so that gradients accumulate back into the parameters
        wp.launch(
            scatter_joint_damping_kernel,
            dim=(self.num_worlds, self.dofs_per_world),
            inputs=[self.damping_param, self.dofs_per_world],
            outputs=[self.model.joint_damping],
        )
        # SolverFeatherstone computes the spatial inertia from body_mass and
        # body_inertia only at construction time; recomputing body_I_m here
        # makes the dynamics differentiable w.r.t. the mass/inertia scales
        wp.launch(
            compute_spatial_inertia_kernel,
            dim=(self.num_worlds, self.bodies_per_world),
            inputs=[
                self.body_mass_ref,
                self.body_inertia_ref,
                self.mass_scale,
                self.inertia_scale,
                self.bodies_per_world,
            ],
            outputs=[self.solver.body_I_m],
        )
        for f in range(self.num_frames):
            for s in range(self.sim_substeps):
                t = f * self.sim_substeps + s
                wp.launch(
                    friction_torque_kernel,
                    dim=(self.num_worlds, self.dofs_per_world),
                    inputs=[self.states[t].joint_qd, self.friction_param, self.dofs_per_world, self.friction_vel_eps],
                    outputs=[self.controls[t].joint_f],
                )
                self.solver.step(self.states[t], self.states[t + 1], self.controls[t], None, self.sim_dt)
            sampled = self.states[(f + 1) * self.sim_substeps].joint_q
            if record_ref:
                wp.copy(self.joint_q_ref[f], sampled)
            if compute_loss:
                wp.launch(
                    trajectory_loss_kernel,
                    dim=self.dof_count,
                    inputs=[sampled, self.joint_q_ref, f, 1.0 / (self.num_frames * self.dof_count)],
                    outputs=[self.loss],
                )

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.forward_backward()

        self.loss_history.append(self.loss.numpy()[0])

        self.opt_torque.step([self.damping_param.grad, self.friction_param.grad])
        self.opt_scale.step([self.mass_scale.grad, self.inertia_scale.grad])
        self.tape.zero()

        # project the parameters back onto their physically plausible ranges
        for param in (self.damping_param, self.friction_param):
            wp.launch(clamp_param_kernel, dim=len(param), inputs=[0.0, 1.0], outputs=[param])
        for param in (self.mass_scale, self.inertia_scale):
            wp.launch(clamp_param_kernel, dim=len(param), inputs=[0.1, 10.0], outputs=[param])

        if self.verbose:
            print(f"train iter: {self.train_iter:4d} loss: {self.loss_history[-1]:.4e}")
            if self.train_iter % 25 == 0:
                self.print_parameters()

        self.train_iter += 1

    def print_parameters(self):
        np.set_printoptions(precision=4, suppress=True)
        print("  damping        true:", self.damping_true, "\n                found:", self.damping_param.numpy())
        print("  friction       true:", self.friction_true, "\n                found:", self.friction_param.numpy())
        print(
            "  mass scale     true:",
            self.mass_scale_true[1:],
            "\n                found:",
            self.mass_scale.numpy()[1:],
        )
        print(
            "  inertia scale  true:",
            self.inertia_scale_true[1:],
            "\n                found:",
            self.inertia_scale.numpy()[1:],
        )

    def gather_ee_trajectories(self):
        # end-effector positions at the sampled frames, per world
        traj = np.empty((self.num_worlds, self.num_frames, 3))
        for f in range(self.num_frames):
            body_q = self.states[f * self.sim_substeps].body_q.numpy()
            for w in range(self.num_worlds):
                traj[w, f] = body_q[w * self.bodies_per_world + self.ee_body, :3]
        return traj

    def render(self):
        if self.viewer.is_paused():
            self.viewer.begin_frame(self.viewer.time)
            self.viewer.end_frame()
            return

        # replay the latest rollout every few training iterations
        if self.frame > 0 and (self.train_iter - 1) % self.render_interval != 0:
            return

        ee_traj = self.gather_ee_trajectories()
        ref_starts = np.concatenate(self.ee_traj_ref[:, :-1])
        ref_ends = np.concatenate(self.ee_traj_ref[:, 1:])
        sim_starts = np.concatenate(ee_traj[:, :-1])
        sim_ends = np.concatenate(ee_traj[:, 1:])
        ref_colors = wp.full(len(ref_starts), wp.vec3(0.2, 0.8, 0.3), dtype=wp.vec3)
        sim_colors = wp.full(len(sim_starts), wp.vec3(1.0, 0.5, 0.1), dtype=wp.vec3)

        for f in range(self.num_frames):
            self.viewer.begin_frame(self.frame * self.frame_dt)
            if self.loss_history:
                self.viewer.log_scalar("/loss", self.loss_history[-1])
            self.viewer.log_state(self.states[f * self.sim_substeps])
            self.viewer.log_lines(
                "/ee_traj_ref",
                wp.array(ref_starts, dtype=wp.vec3),
                wp.array(ref_ends, dtype=wp.vec3),
                ref_colors,
            )
            self.viewer.log_lines(
                "/ee_traj_sim",
                wp.array(sim_starts, dtype=wp.vec3),
                wp.array(sim_ends, dtype=wp.vec3),
                sim_colors,
            )
            self.viewer.end_frame()
            self.frame += 1

    def test_post_step(self):
        assert np.isfinite(self.loss_history[-1]), "loss must stay finite during training"

    def test_final(self):
        loss_history = np.array(self.loss_history)
        assert len(loss_history) >= 2
        assert np.isfinite(loss_history).all()
        assert loss_history[-1] < loss_history[0], "trajectory loss must decrease"
        if self.train_iter >= 80 and self.num_frames >= 30:
            # with enough iterations on the full-length experiment, the
            # well-excited parameters must be recovered accurately
            assert loss_history[-1] < 0.05 * loss_history[0]
            damping_err = np.abs(self.damping_param.numpy() - self.damping_true).max()
            friction_err = np.abs(self.friction_param.numpy() - self.friction_true).max()
            mass_err = np.abs(self.mass_scale.numpy() - self.mass_scale_true).max()
            assert damping_err < 0.005, f"joint damping must be identified, error={damping_err}"
            assert friction_err < 0.01, f"joint friction must be identified, error={friction_err}"
            assert mass_err < 0.3, f"link mass scales must approach the truth, error={mass_err}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--sim-duration", type=float, default=1.0, help="Duration of the recorded trajectory in seconds."
        )
        parser.add_argument(
            "--verbose", action="store_true", help="Print out additional status messages during execution."
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
