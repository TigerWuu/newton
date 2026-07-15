# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot SO-101 Reach (SHAC on differentiable simulation)
#
# Trains the SO-101 reach task with Short-Horizon Actor-Critic (SHAC,
# Xu et al., ICLR 2022, https://short-horizon-actor-critic.github.io/)
# instead of PPO: the policy gradient is computed *analytically* by
# backpropagating rewards through the simulator itself.
#
# The differentiable environment reuses the machinery of the SO-101 system
# identification example (example_diffsim_so101_sysid_true): the model is
# finalized with requires_grad=True, each short-horizon window owns one
# State per substep and one Control per substep (so the backward pass never
# aliases intermediates), and SolverFeatherstone steps are recorded on a
# wp.Tape. A torch.autograd.Function wraps each policy step, so the full
# closed-loop BPTT gradient (through observations, policy, and dynamics)
# is assembled by torch across the window. Window start states are
# detached leaves, per SHAC. Episode length is rounded to a multiple of
# the horizon so resets stay synchronized at window boundaries (the reach
# task terminates on time-out only).
#
# Task definition (observations, reward terms, goals) matches the PPO
# variant in example_robot_so101_reach.py -- the reward is only rescaled
# (x10, no dt factor) to suit SHAC's analytic-gradient regime -- so results
# are comparable and trained policies play back on the fast graph-captured
# (non-differentiable) environment.
#
# Commands:
#   python -m newton.examples robot_so101_reach_shac --train --num-envs 256
#   python -m newton.examples robot_so101_reach_shac
#
###########################################################################

import math
import os

import torch
import warp as wp

import newton
import newton.examples
from newton.examples.rl.example_robot_so101_reach import (
    GOAL_MIN_HEIGHT,
    GOAL_RANGE_XY,
    GOAL_RANGE_Z,
    SO101ReachEnv,
)
from newton.examples.rl.shac import SHACTrainer, load_shac_actor
from newton.examples.rl.so101_rl_env import (
    ExamplePolicyPlayer,
    build_so101_world,
    create_task_parser,
    find_latest_checkpoint,
    force_null_viewer_for_training,
)
from newton.examples.robot.example_robot_so101_digital_twin import friction_torque_kernel

TASK_NAME = "so101_reach_shac"

ACTION_SCALE = 0.5  # joint target = home pose + scale * action [rad]
RESET_JOINT_NOISE = 0.1  # uniform initial joint offset around the home pose [rad]
EPISODE_LENGTH_S = 5.0


class _SO101WindowStep(torch.autograd.Function):
    """One differentiable policy step (``sim_substeps`` solver steps).

    Forward records the substeps of window slot ``step_idx`` on a ``wp.Tape``;
    backward seeds the output adjoints and replays the tape, returning
    gradients w.r.t. the input joint state and the joint position targets.
    Each step owns disjoint State/Control buffers, so per-step tapes compose
    into a full-window BPTT chain orchestrated by torch autograd.
    """

    @staticmethod
    def forward(ctx, joint_q_in, joint_qd_in, joint_target, env, step_idx):
        t0 = step_idx * env.sim_substeps
        states, controls = env.states, env.controls

        if step_idx == 0:
            # load the (detached) persistent state into the window's first
            # state: a constant leaf, so gradients do not chain across windows
            wp.copy(states[0].joint_q, wp.from_torch(joint_q_in.detach().contiguous()))
            wp.copy(states[0].joint_qd, wp.from_torch(joint_qd_in.detach().contiguous()))

        # the target is constant across the substeps of one policy step; write
        # it into every substep's control as a leaf and sum the grads in backward
        target_wp = wp.from_torch(joint_target.detach().contiguous())
        for s in range(env.sim_substeps):
            wp.copy(controls[t0 + s].joint_target_q, target_wp)

        tape = wp.Tape()
        with tape:
            for s in range(env.sim_substeps):
                t = t0 + s
                # swap in this substep's private copies of the solver-level
                # scratch matrices (see _solver_aux in the env constructor)
                for name, arr in env._solver_aux[t].items():
                    setattr(env.solver, name, arr)
                # identified Coulomb friction, same smooth model the parameters
                # were fitted with (differentiable tanh)
                wp.launch(
                    friction_torque_kernel,
                    dim=env.model.joint_dof_count,
                    inputs=[states[t].joint_qd, env.model.joint_friction, env.friction_vel_eps],
                    outputs=[controls[t].joint_f],
                    device=env.sim_device,
                )
                env.solver.step(states[t], states[t + 1], controls[t], None, env.sim_dt)

        ctx.env = env
        ctx.t0 = t0
        ctx.tape = tape

        out = states[t0 + env.sim_substeps]
        return (
            wp.to_torch(out.joint_q).clone(),
            wp.to_torch(out.joint_qd).clone(),
            wp.to_torch(out.body_q).clone(),  # (body_count, 7)
        )

    @staticmethod
    def backward(ctx, grad_q, grad_qd, grad_body_q):
        env, t0, tape = ctx.env, ctx.t0, ctx.tape
        states, controls = env.states, env.controls
        out = states[t0 + env.sim_substeps]

        # seed the output adjoints with the incoming torch gradients
        wp.copy(out.joint_q.grad, wp.from_torch(grad_q.contiguous()))
        wp.copy(out.joint_qd.grad, wp.from_torch(grad_qd.contiguous()))
        wp.copy(out.body_q.grad, wp.from_torch(grad_body_q.contiguous(), dtype=wp.transform))
        tape.backward()

        # per-env adjoint norm clipping: stiff events can still amplify
        # adjoints across the window; rescaling each world's adjoint to a
        # bounded norm truncates runaway BPTT modes while leaving healthy
        # worlds (norms <= 1e-2 here) untouched and direction-preserved
        def _rescale(g: torch.Tensor, max_norm: float = 1.0) -> torch.Tensor:
            g = torch.nan_to_num(g).view(env.num_envs, -1)
            norms = g.norm(dim=-1, keepdim=True)
            env._adjoint_peak = max(env._adjoint_peak, float(norms.max()))  # BPTT health telemetry
            scale = max_norm / norms.clamp_min(max_norm)
            return (g * scale).reshape(-1)

        grad_q_in = _rescale(wp.to_torch(states[t0].joint_q.grad).clone())
        grad_qd_in = _rescale(wp.to_torch(states[t0].joint_qd.grad).clone())
        grad_target = _rescale(
            sum(wp.to_torch(controls[t0 + s].joint_target_q.grad).clone() for s in range(env.sim_substeps))
        )

        # zero the adjoints AFTER backward: Tape.gradients is populated during
        # the backward pass, so a pre-backward zero() on a fresh tape is a
        # no-op and stale adjoints would accumulate across iterations in the
        # reused window buffers
        tape.zero()
        return grad_q_in, grad_qd_in, grad_target, None, None


class SO101ReachDiffEnv:
    """Differentiable batched SO-101 reach environment (SHAC window protocol)."""

    def __init__(self, args, num_envs: int, horizon: int):
        self.num_envs = num_envs
        self.horizon = horizon

        # same control rate and substepping as the digital twin / PPO variant
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(int(args.substeps), 1)
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.friction_vel_eps = 0.1

        self.sim_device = wp.get_device()
        self.device = "cuda" if self.sim_device.is_cuda else "cpu"
        torch.manual_seed(args.seed)
        if self.sim_device.is_cuda:
            # keep torch on Warp's stream: forward views and the autograd
            # backward (which runs on the forward op's stream) stay ordered
            torch.cuda.set_device(self.sim_device.ordinal)
            torch.cuda.set_stream(torch.cuda.ExternalStream(wp.get_stream(self.sim_device).cuda_stream))

        world = newton.ModelBuilder()
        build_so101_world(world, args)
        self.ee_body_local = next(i for i, label in enumerate(world.body_label) if label.endswith("/gripper"))
        shoulder_local = next(i for i, label in enumerate(world.body_label) if label.endswith("/shoulder"))
        self.bodies_per_world = world.body_count

        scene = newton.ModelBuilder()
        scene.replicate(world, num_envs)
        scene.add_ground_plane()
        self.model = scene.finalize(requires_grad=True)

        # soften the joint-limit springs for training: the 1e4 default sits at
        # the explicit-integration stability boundary (omega*dt ~ 1.6 with the
        # identified armature), and every limit contact then amplifies the
        # BPTT adjoints exponentially. The limit stiffness is an unidentified
        # default -- the sysid deliberately kept its data away from the limits
        # -- so a gradient-friendly value does not depart from the twin.
        self.model.joint_limit_ke.fill_(1.0e2)
        self.model.joint_limit_kd.fill_(1.0e1)

        self.dofs_per_world = self.model.joint_dof_count // num_envs
        self.num_actions = self.dofs_per_world
        self.obs_dim = 3 * self.dofs_per_world + 3

        self.solver = newton.solvers.SolverFeatherstone(self.model)

        # one State per substep and one Control per substep, so the tape's
        # backward pass never aliases intermediates (sysid slot pattern)
        n_sub = horizon * self.sim_substeps
        self.states = [self.model.state() for _ in range(n_sub + 1)]
        self.controls = [self.model.control() for _ in range(n_sub)]

        # per-substep copies of the solver-level scratch (mass matrix M,
        # Jacobian J, and factorization P/H/L): SolverFeatherstone recomputes
        # these into *shared* solver attributes every step, so without
        # per-step copies the tape's backward would replay earlier steps
        # against the LAST step's matrices -- gradients would be correct only
        # while the configuration barely changes across the window
        self._solver_aux_names = ("M", "J", "P", "H", "L")
        self._solver_aux = [
            {name: wp.zeros_like(getattr(self.solver, name), requires_grad=True) for name in self._solver_aux_names}
            for _ in range(n_sub)
        ]

        # home-pose FK on a scratch state for goal-region calibration.
        # detach: with requires_grad=True on the model, wp.to_torch views are
        # grad-tracked, and these constants must be autograd leaves -- otherwise
        # every window's graph would chain into this one-time init subgraph
        scratch = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, scratch)
        body_q0 = wp.to_torch(scratch.body_q).detach().view(num_envs, self.bodies_per_world, 7).clone()
        self.home_ee = body_q0[:, self.ee_body_local, :3].clone()
        self.shoulder_pos = body_q0[:, shoulder_local, :3].clone()
        chain = body_q0[0, shoulder_local : self.ee_body_local + 1, :3]
        self.max_reach = 0.95 * torch.linalg.norm(chain[1:] - chain[:-1], dim=-1).sum().item()

        self.default_q = wp.to_torch(self.model.joint_q).detach().view(num_envs, -1)[0].clone()
        # keep targets a margin inside the limits: the joint-limit springs are
        # stiff, and driving into them is the main source of exploding BPTT
        # gradients (the sysid example widens its limits for the same reason)
        limit_margin = 0.05
        self.limit_lower = (
            wp.to_torch(self.model.joint_limit_lower).detach().view(num_envs, -1)[0].clone() + limit_margin
        )
        self.limit_upper = (
            wp.to_torch(self.model.joint_limit_upper).detach().view(num_envs, -1)[0].clone() - limit_margin
        )

        # synchronized time-out resets at window boundaries keep the BPTT
        # chain free of mid-window discontinuities (reach never terminates early)
        self.max_episode_steps = max(1, math.ceil(EPISODE_LENGTH_S * self.fps / horizon)) * horizon
        self.episode_steps = 0
        self.goal_pos = self.home_ee.clone()
        self._last_dist = torch.zeros(num_envs, device=self.device)
        self._adjoint_peak = 0.0

        self.reset_all()

    @torch.no_grad()
    def reset_all(self):
        n = self.num_envs
        q = self.default_q.unsqueeze(0) + RESET_JOINT_NOISE * (
            2.0 * torch.rand(n, self.dofs_per_world, device=self.device) - 1.0
        )
        q = torch.clamp(q, self.limit_lower, self.limit_upper)
        self.q_persist = q.reshape(-1)
        self.qd_persist = torch.zeros_like(self.q_persist)
        self.prev_action_persist = torch.zeros(n, self.num_actions, device=self.device)
        self.episode_steps = 0

        # goal sampling, identical to the PPO reach task
        offset = torch.empty(n, 3, device=self.device)
        offset[:, :2].uniform_(-GOAL_RANGE_XY, GOAL_RANGE_XY)
        offset[:, 2].uniform_(*GOAL_RANGE_Z)
        goal = self.home_ee + offset
        arm = goal - self.shoulder_pos
        dist = torch.linalg.norm(arm, dim=-1, keepdim=True).clamp(min=1e-6)
        goal = torch.where(dist > self.max_reach, self.shoulder_pos + arm * (self.max_reach / dist), goal)
        goal[:, 2].clamp_(min=GOAL_MIN_HEIGHT)
        self.goal_pos = goal

    def _obs(self, q: torch.Tensor, qd: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
        return torch.cat([q - self.default_q, qd, self.goal_pos - self.home_ee, prev_action], dim=-1)

    # --- SHAC window protocol -------------------------------------------------

    def begin_window(self) -> torch.Tensor:
        self._step_idx = 0
        self._q = self.q_persist
        self._qd = self.qd_persist
        self._prev_action = self.prev_action_persist
        n = self.num_envs
        return self._obs(self._q.view(n, -1), self._qd.view(n, -1), self._prev_action)

    def step_diff(self, actions: torch.Tensor):
        n = self.num_envs
        target = torch.clamp(self.default_q + ACTION_SCALE * actions, self.limit_lower, self.limit_upper).reshape(-1)

        q, qd, body_q = _SO101WindowStep.apply(self._q, self._qd, target, self, self._step_idx)
        self._step_idx += 1

        q2, qd2 = q.view(n, -1), qd.view(n, -1)
        ee = body_q.view(n, self.bodies_per_world, 7)[:, self.ee_body_local, :3]
        dist = torch.linalg.norm(ee - self.goal_pos, dim=-1)
        self._last_dist = dist.detach()

        # same IsaacLab-style reach terms as the PPO variant, but scaled up
        # (x10, no dt scaling): SHAC follows analytic reward gradients, and
        # dt-scaled ~1e-4/step rewards would put those gradients two orders
        # of magnitude below the regime the SHAC hyperparameters are tuned
        # for (and below the critic's obs-gradient noise floor early on)
        reward = (
            -2.0 * dist
            + 1.0 * (1.0 - torch.tanh(dist / 0.1))
            - 1.0e-3 * torch.sum(torch.square(actions - self._prev_action), dim=-1)
            - 1.0e-3 * torch.sum(torch.square(qd2), dim=-1)
        )

        obs = self._obs(q2, qd2, actions)
        self._q, self._qd, self._prev_action = q, qd, actions
        return obs, reward

    def end_window(self) -> dict:
        self.q_persist = self._q.detach().clone()
        self.qd_persist = self._qd.detach().clone()
        self.prev_action_persist = self._prev_action.detach().clone()
        self.episode_steps += self.horizon
        # recover from a diverged forward sim instead of poisoning training
        if not (torch.isfinite(self.q_persist).all() and torch.isfinite(self.qd_persist).all()):
            print("warning: non-finite simulation state detected; resetting all environments")
            self.reset_all()
            return {"ee_goal_distance": float("nan"), "adjoint_peak": self._adjoint_peak}
        log = {"ee_goal_distance": self._last_dist.mean().item(), "adjoint_peak": self._adjoint_peak}
        self._adjoint_peak = 0.0
        if self.episode_steps >= self.max_episode_steps:
            self.reset_all()
        return log


if __name__ == "__main__":
    parser = create_task_parser()
    # h=16 (not the paper's 32): the servo arm's BPTT gradients amplify
    # smoothly with depth, and shorter windows keep the product bounded
    parser.add_argument("--horizon", type=int, default=16, help="SHAC window length [policy steps].")
    parser.add_argument("--actor-lr", type=float, default=2.0e-3, help="Actor learning rate.")
    parser.add_argument("--critic-lr", type=float, default=1.0e-3, help="Critic learning rate.")
    parser.set_defaults(logdir=os.path.join("logs", "shac"))
    force_null_viewer_for_training()
    viewer, args = newton.examples.init(parser)

    if args.train:
        from datetime import datetime

        num_envs = args.num_envs if args.num_envs is not None else 256
        env = SO101ReachDiffEnv(args, num_envs, args.horizon)
        log_dir = os.path.join(args.logdir, TASK_NAME, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        trainer = SHACTrainer(
            env,
            horizon=args.horizon,
            # gamma 0.95 (not 0.99): a 5 s servo reach needs no long credit
            # horizon, and the smaller value scale lets the critic reach its
            # bootstrap equilibrium quickly instead of drifting under the actor
            gamma=0.95,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            log_dir=log_dir,
        )
        if args.checkpoint:
            trainer.load(args.checkpoint)
            print(f"resumed from checkpoint {args.checkpoint}")
        trainer.learn(args.max_iterations)
    else:
        # play back on the fast graph-captured (non-differentiable) env; the
        # observation layout matches the training env exactly
        num_envs = args.num_envs if args.num_envs is not None else (8 if args.test else 16)
        env = SO101ReachEnv(args, num_envs)

        policy = None
        checkpoint = args.checkpoint or find_latest_checkpoint(args.logdir, TASK_NAME)
        if checkpoint is None:
            print("no checkpoint found; playing with a zero policy (arm holds the home pose)")
        else:
            actor = load_shac_actor(checkpoint, env.device)
            print(f"loaded SHAC policy from {checkpoint}")

            def policy(obs):
                return actor(obs["policy"], stochastic=False)

        newton.examples.run(ExamplePolicyPlayer(viewer, args, env, policy), args)
