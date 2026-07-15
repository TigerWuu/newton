# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Vectorized SO-101 environment base for RL training with rsl_rl.

Implements the ``rsl_rl.env.VecEnv`` protocol (duck-typed, so rsl_rl is only
required for training, not for playback) on top of a Newton simulation:

* :meth:`newton.ModelBuilder.replicate` stamps out ``num_envs`` worlds that are
  stepped in a single batched solve.
* Observations / actions are exchanged with PyTorch through zero-copy
  ``wp.to_torch`` views of the state arrays. Torch is redirected onto Warp's
  CUDA stream so the two never race.
* Per-world resets scatter randomized joint states into the masked worlds and
  call :meth:`newton.solvers.SolverBase.reset` so solvers with internal
  buffers (e.g. MuJoCo warm-starts) are cleared for exactly those worlds.

Tasks subclass :class:`SO101RslRlEnv` and override the ``_build_*`` and
``_compute_*`` hooks; see ``example_robot_so101_reach.py`` for a complete task.

rsl_rl (BSD-3-Clause) is vendored at ``newton/_src/rl/rsl_rl`` so it can be
modified in-repo; this module prepends that directory to ``sys.path`` so
``import rsl_rl`` resolves to the vendored copy. Its runtime dependencies are
not vendored: ``uv pip install tensordict gitpython tensorboard``.
"""

import math
import os
import sys

import torch
import warp as wp

import newton
import newton.examples

# prefer the vendored rsl_rl over any installed copy
_VENDORED_RSL_RL = os.path.join(os.path.dirname(newton.__file__), "_src", "rl")
if os.path.isdir(os.path.join(_VENDORED_RSL_RL, "rsl_rl")) and _VENDORED_RSL_RL not in sys.path:
    sys.path.insert(0, _VENDORED_RSL_RL)

# reuse the identified-parameter tooling and the smoothed Coulomb friction
# model from the digital twin so trained policies see the same dynamics that
# were fit to the real hardware
from newton.examples.robot.example_robot_so101_digital_twin import (
    apply_identified_parameters,
    find_latest_identified_csv,
    friction_torque_kernel,
    load_identified_parameters,
)

wp.set_module_options({"enable_backward": False})

try:
    from tensordict import TensorDict
except ImportError:
    TensorDict = None  # playback with a zero policy works without tensordict

# rest pose of the arm ("ready" configuration), same as the digital twin
HOME_POSE = {
    "Rotation": -0.2736,
    "Pitch": -0.6109,
    "Elbow": -0.0745,
    "Wrist_Pitch": 1.5148,
    "Wrist_Roll": -1.6034,
    "Jaw": -0.1465,
}


def build_so101_world(builder: newton.ModelBuilder, args):
    """Populate one world: the SO-101 arm at its home pose, optionally with identified parameters."""
    builder.add_usd(
        newton.examples.get_asset("so101.usd"),
        enable_self_collisions=False,
        collapse_fixed_joints=False,
        hide_collision_shapes=True,
    )

    # optionally overwrite the CAD drives/inertia with values identified from
    # the real arm (see example_diffsim_so101_sysid_true) so trained policies
    # transfer to the hardware the twin was fit to
    params_path = args.params or find_latest_identified_csv(args.plot_dir)
    if params_path and os.path.exists(params_path):
        joint_params, link_params = load_identified_parameters(params_path)
        apply_identified_parameters(builder, joint_params, link_params, args.armature)
        print(f"loaded identified parameters from {params_path}")
    else:
        for joint_idx in range(len(builder.joint_label)):
            builder.joint_armature[builder.joint_q_start[joint_idx]] = args.armature
        print("no identified_parameters.csv found; using nominal USD parameters")

    for joint_idx, label in enumerate(builder.joint_label):
        name = label.rsplit("/", 1)[-1]
        if name in HOME_POSE:
            builder.joint_q[builder.joint_q_start[joint_idx]] = HOME_POSE[name]


class SO101RslRlEnv:
    """Batched SO-101 arm environment implementing the rsl_rl ``VecEnv`` protocol."""

    # task defaults, overridable by subclasses
    episode_length_s = 5.0
    action_scale = 0.5  # joint target = home pose + action_scale * action [rad]
    reset_joint_noise = 0.1  # uniform initial joint offset around the home pose [rad]

    def __init__(self, args, num_envs: int, spacing: tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self.num_envs = num_envs
        self.spacing = spacing

        # match the digital twin: 50 Hz control, identified params fit at 4 substeps
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(int(args.substeps), 2)
        # the captured substep loop swaps state_0/state_1 in Python; an even
        # count keeps state_0 pointing at the same buffers after each launch,
        # which the persistent torch views below rely on
        if self.sim_substeps % 2 != 0:
            self.sim_substeps += 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.friction_vel_eps = 0.1  # same smoothing width as the identification

        self.sim_device = wp.get_device()
        self.device = "cuda" if self.sim_device.is_cuda else "cpu"  # rsl_rl VecEnv attr

        torch.manual_seed(args.seed)
        if self.sim_device.is_cuda:
            # Warp launches on its own CUDA stream while torch defaults to the
            # legacy stream; without this, zero-copy views race with the solver
            torch.cuda.set_device(self.sim_device.ordinal)
            torch.cuda.set_stream(torch.cuda.ExternalStream(wp.get_stream(self.sim_device).cuda_stream))

        # --- build one world, then replicate it -----------------------------
        world = newton.ModelBuilder()
        self._build_world(world, args)

        # per-world entity indices resolved from labels before replication
        self.ee_body_local = next(i for i, label in enumerate(world.body_label) if label.endswith("/gripper"))
        self.shoulder_body_local = next(i for i, label in enumerate(world.body_label) if label.endswith("/shoulder"))
        self.bodies_per_world = world.body_count
        self.particles_per_world = world.particle_count

        scene = newton.ModelBuilder()
        self._build_scene(scene, world, args)
        self.model = scene.finalize()

        self.coords_per_world = self.model.joint_coord_count // num_envs
        self.dofs_per_world = self.model.joint_dof_count // num_envs
        self.num_actions = self.dofs_per_world

        self._create_solver(args)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # --- persistent zero-copy torch views of the simulation state -------
        # all SO-101 joints are 1-dof revolute, so coord and dof views align
        self.q = wp.to_torch(self.state_0.joint_q).view(num_envs, self.coords_per_world)
        self.qd = wp.to_torch(self.state_0.joint_qd).view(num_envs, self.dofs_per_world)
        self.body_q = wp.to_torch(self.state_0.body_q).view(num_envs, self.bodies_per_world, 7)
        self.joint_target = wp.to_torch(self.control.joint_target_q).view(num_envs, self.coords_per_world)

        self.default_q = wp.to_torch(self.model.joint_q).view(num_envs, -1)[0].clone()
        limit_lower = wp.to_torch(self.model.joint_limit_lower).view(num_envs, -1)[0].clone()
        limit_upper = wp.to_torch(self.model.joint_limit_upper).view(num_envs, -1)[0].clone()
        self.joint_limit_lower = limit_lower
        self.joint_limit_upper = limit_upper

        # --- rsl_rl VecEnv bookkeeping ---------------------------------------
        self.max_episode_length = math.ceil(self.episode_length_s / self.frame_dt)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.actions = torch.zeros(num_envs, self.num_actions, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.cfg = {"num_envs": num_envs, "episode_length_s": self.episode_length_s}

        # boolean world mask consumed by Solver.reset (clears per-world solver
        # internals, e.g. MuJoCo warm-starts; a no-op for SolverFeatherstone)
        self._world_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._world_mask_wp = wp.from_torch(self._world_mask, dtype=wp.bool)

        self._init_task(args)
        self._reset_idx(torch.arange(num_envs, device=self.device))

        # capture the substep loop in a CUDA graph; per-step inputs
        # (joint targets) are written into fixed buffers before each launch
        self.use_cuda_graph = self.sim_device.is_cuda and self._supports_cuda_graph()
        self.graph = None
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self._physics_substeps()
            self.graph = capture.graph

    # --- build hooks ---------------------------------------------------------

    def _build_world(self, builder: newton.ModelBuilder, args):
        """Populate one world: the SO-101 arm, optionally with identified parameters."""
        build_so101_world(builder, args)

    def _build_scene(self, scene: newton.ModelBuilder, world: newton.ModelBuilder, args):
        """Replicate the world and add shared entities (ground plane)."""
        scene.replicate(world, self.num_envs, spacing=self.spacing)
        scene.add_ground_plane()

    def _create_solver(self, args):
        # SolverFeatherstone matches the conditions the SO-101 parameters were
        # identified under (same solver and substep count as the digital twin)
        self.solver = newton.solvers.SolverFeatherstone(self.model)

    def _supports_cuda_graph(self) -> bool:
        return True

    # --- task hooks ------------------------------------------------------------

    def _init_task(self, args):
        raise NotImplementedError

    def _reset_task(self, env_ids: torch.Tensor):
        """Resample goals / task state for the given worlds."""
        raise NotImplementedError

    def _compute_obs(self) -> torch.Tensor:
        raise NotImplementedError

    def _compute_rewards(self) -> torch.Tensor:
        raise NotImplementedError

    def _compute_dones(self) -> torch.Tensor:
        # default: fixed-length episodes (time-out only, like IsaacLab reach)
        return self.episode_length_buf >= self.max_episode_length

    def _log_dict(self) -> dict:
        return {}

    # --- physics ----------------------------------------------------------------

    def _apply_actions(self):
        target = self.default_q + self.action_scale * self.actions
        torch.clamp(target, self.joint_limit_lower, self.joint_limit_upper, out=self.joint_target)

    def _physics_substeps(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # identified Coulomb friction as a generalized force, recomputed
            # from the current joint velocity each substep (Featherstone does
            # not read Model.joint_friction; matches the identification setup)
            wp.launch(
                friction_torque_kernel,
                dim=self.model.joint_dof_count,
                inputs=[self.state_0.joint_qd, self.model.joint_friction, self.friction_vel_eps],
                outputs=[self.control.joint_f],
                device=self.sim_device,
            )

            # contacts=None: the reach task never needs arm-ground collision
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _step_physics(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._physics_substeps()

    # --- rsl_rl VecEnv API ----------------------------------------------------

    def get_observations(self):
        obs = {"policy": self._compute_obs()}
        if TensorDict is not None:
            return TensorDict(obs, batch_size=[self.num_envs])
        return obs

    def step(self, actions: torch.Tensor):
        self.prev_actions.copy_(self.actions)
        self.actions.copy_(actions.to(self.device))
        self._apply_actions()

        self._step_physics()

        self.episode_length_buf += 1
        time_outs = self.episode_length_buf >= self.max_episode_length
        dones = self._compute_dones()
        rewards = self._compute_rewards()

        env_ids = dones.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._reset_idx(env_ids)

        extras = {"time_outs": time_outs, "log": self._log_dict()}
        return self.get_observations(), rewards, dones, extras

    def reset(self):
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.get_observations()

    def _reset_idx(self, env_ids: torch.Tensor):
        n = len(env_ids)
        q = self.default_q.unsqueeze(0) + self.reset_joint_noise * (
            2.0 * torch.rand(n, self.coords_per_world, device=self.device) - 1.0
        )
        q = torch.clamp(q, self.joint_limit_lower, self.joint_limit_upper)
        self.q[env_ids] = q
        self.qd[env_ids] = 0.0
        self.joint_target[env_ids] = q  # hold the sampled pose until the first action

        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0

        # clear per-world solver internals (no-op for Featherstone; clears
        # warm-starts/ctrl for SolverMuJoCo). flags=0: we scatter randomized
        # joint states ourselves instead of restoring the model defaults.
        self._world_mask.zero_()
        self._world_mask[env_ids] = True
        self.solver.reset(self.state_0, world_mask=self._world_mask_wp, flags=0)

        # refresh body poses of the reset worlds (one articulation per world)
        indices = wp.from_torch(env_ids.to(dtype=torch.int32).contiguous(), dtype=wp.int32)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0, indices=indices)

        self._reset_task(env_ids)

    # --- shared helpers ---------------------------------------------------------

    @property
    def ee_pos(self) -> torch.Tensor:
        """End-effector (gripper body origin) positions, shape (num_envs, 3)."""
        return self.body_q[:, self.ee_body_local, :3]


def rsl_rl_ppo_cfg() -> dict:
    """Default PPO training configuration (IsaacLab reach-style hyperparameters)."""
    return {
        "num_steps_per_env": 24,
        "save_interval": 50,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "logger": "tensorboard",
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.005,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [64, 64],
            "activation": "elu",
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0},
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [64, 64],
            "activation": "elu",
        },
    }


def find_latest_checkpoint(logdir: str, task_name: str) -> str | None:
    """Return the newest ``model_*.pt`` under ``logdir/task_name``, if any."""
    import glob  # noqa: PLC0415

    matches = glob.glob(os.path.join(logdir, task_name, "*", "model_*.pt"))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def train(env, args, task_name: str):
    """Run rsl_rl PPO training on the given environment."""
    from datetime import datetime  # noqa: PLC0415

    from rsl_rl.runners import OnPolicyRunner  # noqa: PLC0415

    log_dir = os.path.join(args.logdir, task_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(env, rsl_rl_ppo_cfg(), log_dir=log_dir, device=env.device)
    if args.checkpoint:
        runner.load(args.checkpoint)
        print(f"resumed from checkpoint {args.checkpoint}")

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    save_path = os.path.join(log_dir, f"model_{runner.current_learning_iteration + 1}.pt")
    runner.save(save_path)
    print(f"training finished; final checkpoint: {save_path}")


def load_policy(env, args, task_name: str):
    """Load the inference policy from a checkpoint, or None for a zero policy."""
    checkpoint = args.checkpoint or find_latest_checkpoint(args.logdir, task_name)
    if checkpoint is None:
        print("no checkpoint found; playing with a zero policy (arm holds the home pose)")
        return None

    from rsl_rl.runners import OnPolicyRunner  # noqa: PLC0415

    runner = OnPolicyRunner(env, rsl_rl_ppo_cfg(), log_dir=None, device=env.device)
    runner.load(checkpoint)
    print(f"loaded policy from {checkpoint}")
    return runner.get_inference_policy(env.device)


class ExamplePolicyPlayer:
    """Newton ``Example``-format wrapper that plays a (trained or zero) policy."""

    def __init__(self, viewer, args, env, policy):
        self.viewer = viewer
        self.env = env
        self.policy = policy
        self.frame_dt = env.frame_dt
        self.sim_time = 0.0
        # expose the state for the standard NaN checks in newton.examples.run
        self.model = env.model
        self.state = env.state_0

        self.viewer.set_model(env.model)
        if env.spacing == (0.0, 0.0, 0.0) and hasattr(self.viewer, "set_world_offsets"):
            # worlds are simulated at the origin; separate them visually only
            self.viewer.set_world_offsets((0.6, 0.6, 0.0))
        self.viewer.set_camera(pos=wp.vec3(0.9, 0.9, 0.6), pitch=-20.0, yaw=-135.0)

    def step(self):
        obs = self.env.get_observations()
        if self.policy is not None:
            with torch.inference_mode():
                actions = self.policy(obs)
        else:
            actions = torch.zeros(self.env.num_envs, self.env.num_actions, device=self.env.device)
        self.env.step(actions)
        self.state = self.env.state_0
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.env.state_0)
        self.viewer.end_frame()

    def test_final(self):
        obs = self.env._compute_obs()
        assert torch.isfinite(obs).all(), "observations must be finite"
        q = self.env.q
        eps = 1e-3
        assert ((q >= self.env.joint_limit_lower - eps) & (q <= self.env.joint_limit_upper + eps)).all(), (
            "joint positions must stay within limits"
        )


def create_task_parser():
    """Base example parser extended with RL training options."""
    parser = newton.examples.create_parser()
    parser.add_argument("--num-envs", type=int, default=None, help="Number of parallel worlds (task default if unset).")
    parser.add_argument("--train", action="store_true", default=False, help="Train with rsl_rl PPO (headless).")
    parser.add_argument("--max-iterations", type=int, default=300, help="PPO learning iterations.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to resume training / play from.")
    parser.add_argument("--logdir", type=str, default=os.path.join("logs", "rsl_rl"), help="Training log directory.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Path to an identified_parameters.csv (default: latest under --plot-dir).",
    )
    parser.add_argument("--plot-dir", type=str, default="logs/sysid", help="Directory searched for identified parameters.")
    parser.add_argument("--armature", type=float, default=0.1, help="Reflected servo rotor inertia.")
    parser.add_argument(
        "--substeps", type=int, default=4, help="Physics substeps per control step (rounded up to even)."
    )
    return parser


def force_null_viewer_for_training():
    """Default --viewer to null when --train is passed (called before examples.init)."""
    import sys  # noqa: PLC0415

    if "--train" in sys.argv and "--viewer" not in sys.argv:
        sys.argv += ["--viewer", "null"]
