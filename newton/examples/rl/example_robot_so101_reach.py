# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot SO-101 Reach (RL training with rsl_rl)
#
# Trains a PPO policy that moves the SO-101 gripper to randomly sampled
# Cartesian goal positions. The environment is a batch of `--num-envs`
# replicated Newton worlds stepped by SolverFeatherstone (the same solver
# the SO-101 digital twin was identified with; if an
# identified_parameters.csv from the sysid example is found it is applied
# automatically, so trained policies match the real hardware dynamics).
#
# Observations, rewards, and hyperparameters mirror the IsaacLab
# Isaac-Reach task family:
#   obs    = [joint_pos - home, joint_vel, goal command, last action]
#   reward = -0.2*|ee-goal| + 0.1*(1-tanh(|ee-goal|/0.1))
#            - 1e-4*|action rate|^2 - 1e-4*|joint_vel|^2   (all * step dt)
# The 5-DOF arm cannot track a full 6-DOF pose, so unlike IsaacLab's
# Franka reach the goal is position-only (no orientation term).
#
# Training uses the rsl_rl copy vendored at newton/_src/rl/rsl_rl and its
# (non-vendored) dependencies:
#   uv pip install tensordict gitpython tensorboard
#
# Commands:
#   # train headless, then watch the result
#   python -m newton.examples robot_so101_reach --train --num-envs 1024
#   python -m newton.examples robot_so101_reach
#   # resume / play a specific checkpoint
#   python -m newton.examples robot_so101_reach --train --checkpoint logs/rsl_rl/so101_reach/<run>/model_300.pt
#   python -m newton.examples robot_so101_reach --checkpoint logs/rsl_rl/so101_reach/<run>/model_300.pt
#
###########################################################################

import torch

import newton.examples
from newton.examples.rl.so101_rl_env import (
    ExamplePolicyPlayer,
    SO101RslRlEnv,
    create_task_parser,
    force_null_viewer_for_training,
    load_policy,
    train,
)

TASK_NAME = "so101_reach"

# goal sampling box around the home end-effector position [m]
GOAL_RANGE_XY = 0.15
GOAL_RANGE_Z = (-0.10, 0.12)
GOAL_MIN_HEIGHT = 0.05


class SO101ReachEnv(SO101RslRlEnv):
    """Reach task: drive the gripper to a random goal position."""

    episode_length_s = 5.0

    def _init_task(self, args):
        # self-calibrating goal region: sample around the home-pose gripper
        # position, clamped into the arm's reach sphere (same construction as
        # the digital twin's teleop target limit)
        self.home_ee = self.ee_pos.clone()
        self.shoulder_pos = self.body_q[:, self.shoulder_body_local, :3].clone()

        chain = self.body_q[0, self.shoulder_body_local : self.ee_body_local + 1, :3]
        self.max_reach = 0.95 * torch.linalg.norm(chain[1:] - chain[:-1], dim=-1).sum().item()

        self.goal_pos = self.home_ee.clone()
        self._ee_goal_dist = torch.zeros(self.num_envs, device=self.device)

    def _reset_task(self, env_ids: torch.Tensor):
        n = len(env_ids)
        offset = torch.empty(n, 3, device=self.device)
        offset[:, :2].uniform_(-GOAL_RANGE_XY, GOAL_RANGE_XY)
        offset[:, 2].uniform_(*GOAL_RANGE_Z)
        goal = self.home_ee[env_ids] + offset

        # keep goals inside the reach sphere so they are always attainable
        arm = goal - self.shoulder_pos[env_ids]
        dist = torch.linalg.norm(arm, dim=-1, keepdim=True).clamp(min=1e-6)
        goal = torch.where(dist > self.max_reach, self.shoulder_pos[env_ids] + arm * (self.max_reach / dist), goal)
        goal[:, 2].clamp_(min=GOAL_MIN_HEIGHT)
        self.goal_pos[env_ids] = goal

    def _compute_obs(self) -> torch.Tensor:
        # IsaacLab reach observation layout (position-only command)
        return torch.cat(
            [
                self.q - self.default_q,
                self.qd,
                self.goal_pos - self.home_ee,  # goal command in the per-world home frame
                self.actions,
            ],
            dim=-1,
        )

    def _compute_rewards(self) -> torch.Tensor:
        dist = torch.linalg.norm(self.ee_pos - self.goal_pos, dim=-1)
        self._ee_goal_dist = dist

        # IsaacLab reach reward terms and weights (RewardManager scales by dt)
        reward = (
            -0.2 * dist
            + 0.1 * (1.0 - torch.tanh(dist / 0.1))
            - 1.0e-4 * torch.sum(torch.square(self.actions - self.prev_actions), dim=-1)
            - 1.0e-4 * torch.sum(torch.square(self.qd), dim=-1)
        )
        return reward * self.frame_dt

    def _log_dict(self) -> dict:
        return {"/metrics/ee_goal_distance": self._ee_goal_dist}


if __name__ == "__main__":
    parser = create_task_parser()
    force_null_viewer_for_training()
    viewer, args = newton.examples.init(parser)

    if args.num_envs is not None:
        num_envs = args.num_envs
    elif args.train:
        num_envs = 1024
    elif args.test:
        num_envs = 8
    else:
        num_envs = 16

    env = SO101ReachEnv(args, num_envs)

    if args.train:
        train(env, args, TASK_NAME)
    else:
        policy = None
        try:
            policy = load_policy(env, args, TASK_NAME)
        except ImportError as exc:
            print(f"rsl_rl unavailable ({exc}); playing with a zero policy")
        newton.examples.run(ExamplePolicyPlayer(viewer, args, env, policy), args)
