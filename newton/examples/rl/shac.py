# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Short-Horizon Actor-Critic (SHAC) on differentiable Newton simulation.

Implements the algorithm from "Accelerated Policy Learning with Parallel
Differentiable Simulation" (Xu et al., ICLR 2022,
https://short-horizon-actor-critic.github.io/):

* The actor is trained with **analytic first-order gradients** obtained by
  backpropagating through short simulation windows (``horizon`` policy steps)
  of a differentiable environment:

      L_actor = -1/(N*h) * sum_envs [ sum_t gamma^t r_t + gamma^h V(s_h) ]

  The gradient flows through the rewards *and* the terminal value via the
  simulator dynamics (BPTT), in contrast to model-free estimators like PPO.
* The critic is trained on TD(lambda) targets computed from the (detached)
  window trajectory, regressed with minibatch MSE against a **target critic**
  that is Polyak-mixed after every iteration to stabilize the bootstrap.
* Window start states are detached leaves, keeping gradients bounded.

The environment must provide the differentiable-window protocol::

    obs = env.begin_window()  # (num_envs, obs_dim), start of window
    obs, reward = env.step_diff(a)  # differentiable through the simulator
    log_dict = env.end_window()  # persist state, handle resets

plus ``num_envs``, ``num_actions``, ``obs_dim``, and ``device`` attributes.
See ``example_robot_so101_reach_shac.py`` for a Newton implementation based
on ``wp.Tape`` and a ``torch.autograd.Function`` simulation bridge.
"""

import os
import time

import torch
from torch import nn


class ActorSHAC(nn.Module):
    """Gaussian policy: MLP mean + state-independent learnable log-std.

    Actions are drawn with the reparameterization trick so the simulation
    gradient flows through the sampling noise.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims=(64, 64), init_log_std: float = -1.0):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ELU()]
            last = h
        head = nn.Linear(last, action_dim)
        # near-zero initial policy: start at the home pose and move away
        # gently, so early windows stay in the well-conditioned gradient regime
        with torch.no_grad():
            head.weight.mul_(0.01)
            head.bias.zero_()
        layers += [head]
        self.mu = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))

    def forward(self, obs: torch.Tensor, stochastic: bool = True) -> torch.Tensor:
        # tanh squash: bounds the action AND saturates the policy Jacobian,
        # keeping the closed-loop (dynamics x policy) BPTT gain bounded --
        # unsquashed policies make the true gradient explode as weights grow
        mu = self.mu(obs)
        if not stochastic:
            return torch.tanh(mu)
        std = torch.exp(self.log_std.clamp(-5.0, 2.0))
        return torch.tanh(mu + std * torch.randn_like(mu))  # squashed rsample


class CriticSHAC(nn.Module):
    """State-value MLP."""

    def __init__(self, obs_dim: int, hidden_dims=(64, 64)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ELU()]
            last = h
        layers += [nn.Linear(last, 1)]
        self.v = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs).squeeze(-1)


def td_lambda_targets(rewards: torch.Tensor, values: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    """TD(lambda) targets over a window.

    Args:
        rewards: shape (horizon, num_envs).
        values: target-critic values, shape (horizon + 1, num_envs).
        gamma: discount factor.
        lam: TD(lambda) mixing factor.

    Returns:
        Targets for V(s_0..s_{h-1}), shape (horizon, num_envs).
    """
    horizon = rewards.shape[0]
    targets = torch.empty_like(rewards)
    g = values[-1]
    for t in reversed(range(horizon)):
        g = rewards[t] + gamma * ((1.0 - lam) * values[t + 1] + lam * g)
        targets[t] = g
    return targets


class SHACTrainer:
    """SHAC training loop on a differentiable-window environment."""

    def __init__(
        self,
        env,
        horizon: int = 32,
        gamma: float = 0.99,
        lam: float = 0.95,
        actor_lr: float = 2.0e-3,
        critic_lr: float = 1.0e-3,
        critic_epochs: int = 8,
        critic_minibatches: int = 4,
        target_critic_alpha: float = 0.4,
        max_grad_norm: float = 1.0,
        hidden_dims=(64, 64),
        init_log_std: float = -2.0,
        log_dir: str | None = None,
        save_interval: int = 50,
    ):
        self.env = env
        self.horizon = horizon
        self.gamma = gamma
        self.lam = lam
        self.critic_epochs = critic_epochs
        self.critic_minibatches = critic_minibatches
        self.target_critic_alpha = target_critic_alpha
        self.max_grad_norm = max_grad_norm
        self.device = env.device
        self.log_dir = log_dir
        self.save_interval = save_interval
        self.iteration = 0

        self.actor = ActorSHAC(env.obs_dim, env.num_actions, hidden_dims, init_log_std).to(self.device)
        self.critic = CriticSHAC(env.obs_dim, hidden_dims).to(self.device)
        self.target_critic = CriticSHAC(env.obs_dim, hidden_dims).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.requires_grad_(False)

        # betas from the SHAC reference implementation (NVlabs/DiffRL)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, betas=(0.7, 0.95))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, betas=(0.7, 0.95))

        self.writer = None
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            try:
                from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415

                self.writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                print("tensorboard not available; logging to stdout only")

    def learn(self, num_iterations: int):
        n, h = self.env.num_envs, self.horizon
        for _ in range(num_iterations):
            t_start = time.time()

            # --- differentiable short-horizon rollout (actor update) --------
            obs = self.env.begin_window()
            obs_seq = [obs.detach()]
            rewards_seq = []
            reward_acc = torch.zeros(n, device=self.device)
            gamma_pow = 1.0
            for _t in range(h):
                actions = self.actor(obs, stochastic=True)
                obs, rewards = self.env.step_diff(actions)
                reward_acc = reward_acc + gamma_pow * rewards
                gamma_pow *= self.gamma
                obs_seq.append(obs.detach())
                rewards_seq.append(rewards.detach())

            # terminal value bootstrap; grads flow into the simulator through
            # the critic *input* only (target critic params are frozen)
            terminal_value = self.target_critic(obs)
            actor_loss = -(reward_acc + gamma_pow * terminal_value).mean() / h

            self.actor_opt.zero_grad()
            actor_loss.backward()
            for p in self.actor.parameters():
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_opt.step()

            log_dict = self.env.end_window()

            # --- critic update on the detached trajectory --------------------
            obs_stack = torch.stack(obs_seq)  # (h + 1, n, obs_dim)
            rewards_stack = torch.stack(rewards_seq)  # (h, n)
            with torch.no_grad():
                values = self.target_critic(obs_stack.view((h + 1) * n, -1)).view(h + 1, n)
                targets = td_lambda_targets(rewards_stack, values, self.gamma, self.lam)
            flat_obs = obs_stack[:h].reshape(h * n, -1)
            flat_targets = targets.reshape(h * n)

            critic_loss_total = 0.0
            batch_size = (h * n) // self.critic_minibatches
            for _epoch in range(self.critic_epochs):
                perm = torch.randperm(h * n, device=self.device)
                for b in range(self.critic_minibatches):
                    idx = perm[b * batch_size : (b + 1) * batch_size]
                    critic_loss = nn.functional.mse_loss(self.critic(flat_obs[idx]), flat_targets[idx])
                    self.critic_opt.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_opt.step()
                    critic_loss_total += critic_loss.item()
            critic_loss_mean = critic_loss_total / (self.critic_epochs * self.critic_minibatches)

            # Polyak-mix the target critic (alpha = fraction of old weights kept)
            with torch.no_grad():
                for p_t, p in zip(self.target_critic.parameters(), self.critic.parameters(), strict=True):
                    p_t.mul_(self.target_critic_alpha).add_((1.0 - self.target_critic_alpha) * p)

            # --- bookkeeping --------------------------------------------------
            self.iteration += 1
            it = self.iteration
            mean_reward = rewards_stack.mean().item()
            dt_iter = time.time() - t_start
            sps = int(n * h / dt_iter)
            extra = " ".join(f"{k}: {v:.4f}" for k, v in (log_dict or {}).items())
            print(
                f"[SHAC it {it:4d}] reward/step: {mean_reward:+.5f}  actor loss: {actor_loss.item():+.5f}  "
                f"critic loss: {critic_loss_mean:.5f}  |grad|: {float(grad_norm):.3f}  {extra}  ({sps} steps/s)"
            )
            if self.writer is not None:
                self.writer.add_scalar("shac/reward_per_step", mean_reward, it)
                self.writer.add_scalar("shac/actor_loss", actor_loss.item(), it)
                self.writer.add_scalar("shac/critic_loss", critic_loss_mean, it)
                self.writer.add_scalar("shac/actor_grad_norm", float(grad_norm), it)
                for k, v in (log_dict or {}).items():
                    self.writer.add_scalar(f"shac/{k}", v, it)

            if self.log_dir is not None and it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.iteration}.pt"))

    def save(self, path: str):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "iteration": self.iteration,
                "obs_dim": self.env.obs_dim,
                "num_actions": self.env.num_actions,
            },
            path,
        )
        print(f"saved checkpoint: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.target_critic.load_state_dict(ckpt["target_critic"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.iteration = ckpt["iteration"]


def load_shac_actor(path: str, device: str, hidden_dims=(64, 64)) -> ActorSHAC:
    """Load just the actor from a SHAC checkpoint, for playback."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    actor = ActorSHAC(ckpt["obs_dim"], ckpt["num_actions"], hidden_dims).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor
