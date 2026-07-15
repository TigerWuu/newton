# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot SO-101 Cloth (RL training with rsl_rl)
#
# Trains a PPO policy where the SO-101 arm drags a small square of cloth
# lying on the ground to a randomly sampled target spot. Each of the
# `--num-envs` replicated worlds couples two solvers, following the
# cloth_franka example: SolverFeatherstone steps the arm dynamically
# (PD joint drives, identified SO-101 parameters if available) while
# SolverVBD steps the cloth, coupled one-way through the collision
# pipeline's cloth-body contacts.
#
# Reward terms are adapted from the IsaacLab Isaac-Lift task family
# (reach-object + object-goal tracking + fine-grained tracking, with
# action-rate and joint-velocity penalties), with the lift bonus replaced
# by planar goal tracking since the scaffold gripper does not grasp:
#   r = 1.0*(1-tanh(|ee-cloth|/0.1)) + 16*(1-tanh(|cloth-goal|/0.15))
#       + 5*(1-tanh(|cloth-goal|/0.05)) - 1e-4*|action rate|^2
#       - 1e-4*|joint_vel|^2                              (all * step dt)
#
# Unlike the reach task, worlds are replicated with physical spacing:
# the particle contact structures are global, so overlapping cloths from
# different worlds would collide with each other at the origin.
#
# Cloth simulation is expensive; start with a few hundred environments.
#
# Training uses the rsl_rl copy vendored at newton/_src/rl/rsl_rl and its
# (non-vendored) dependencies:
#   uv pip install tensordict gitpython tensorboard
#
# Commands:
#   python -m newton.examples robot_so101_cloth --train --num-envs 256
#   python -m newton.examples robot_so101_cloth
#
###########################################################################

import torch
import warp as wp

import newton
import newton.examples
from newton.examples.rl.so101_rl_env import (
    ExamplePolicyPlayer,
    SO101RslRlEnv,
    create_task_parser,
    force_null_viewer_for_training,
    friction_torque_kernel,
    load_policy,
    train,
)

TASK_NAME = "so101_cloth"

# cloth square on the ground in front of the arm (home gripper xy is ~(0.15, 0.06),
# reach sphere radius ~0.36 around the shoulder at ~(0.02, 0.02, 0.09))
CLOTH_CENTER = (0.20, 0.06)
CLOTH_DIM = 16  # 17x17 particles
CLOTH_CELL = 0.014  # -> 0.224 m square
CLOTH_DROP_HEIGHT = 0.01

GOAL_RADIUS_RANGE = (0.08, 0.15)
GOAL_MAX_FROM_SHOULDER = 0.30  # keep goals inside the arm's reach disc [m]


class SO101ClothEnv(SO101RslRlEnv):
    """Cloth-drag task: push/drag the cloth so its center reaches a goal spot."""

    episode_length_s = 6.0

    def __init__(self, args, num_envs: int):
        # VBD needs a finer step than the rigid-only reach task (cloth_franka
        # uses 10 substeps); physical spacing isolates the per-world cloths
        args.substeps = max(args.substeps, 10)
        super().__init__(args, num_envs, spacing=(1.0, 1.0, 0.0))

    # --- build -----------------------------------------------------------------

    def _build_world(self, builder: newton.ModelBuilder, args):
        super()._build_world(builder, args)

        half = 0.5 * CLOTH_DIM * CLOTH_CELL
        builder.add_cloth_grid(
            pos=wp.vec3(CLOTH_CENTER[0] - half, CLOTH_CENTER[1] - half, CLOTH_DROP_HEIGHT),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=CLOTH_DIM,
            dim_y=CLOTH_DIM,
            cell_x=CLOTH_CELL,
            cell_y=CLOTH_CELL,
            mass=1.0e-4,  # ~29 g total, similar areal density to the franka shirt
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=1.0e-1,
            edge_ke=1.0,
            edge_kd=0.0,
            particle_radius=0.006,
        )

    def _build_scene(self, scene: newton.ModelBuilder, world: newton.ModelBuilder, args):
        super()._build_scene(scene, world, args)
        scene.color()  # graph coloring required by SolverVBD

    def _create_solver(self, args):
        # mass matrix refresh once per control step, as in cloth_franka
        self.solver = newton.solvers.SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self.cloth_solver = newton.solvers.SolverVBD(
            self.model,
            iterations=5,
            integrate_with_external_rigid_solver=True,
            particle_self_contact_radius=0.003,
            particle_self_contact_margin=0.003,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.005,
            particle_enable_self_contact=True,
            particle_vertex_contact_buffer_size=16,
            particle_edge_contact_buffer_size=20,
            particle_collision_detection_interval=-1,
        )
        # explicit pipeline for cloth-body contacts with a custom margin
        self.collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_margin=0.01)
        self.contacts = self.collision_pipeline.contacts()

    def _supports_cuda_graph(self) -> bool:
        # the per-step BVH rebuild is host-side logic (cloth_franka does not
        # capture either)
        return False

    # --- physics: Featherstone arm + VBD cloth, cloth_franka coupling ----------

    def _physics_substeps(self):
        self.cloth_solver.rebuild_bvh(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            wp.launch(
                friction_torque_kernel,
                dim=self.model.joint_dof_count,
                inputs=[self.state_0.joint_qd, self.model.joint_friction, self.friction_vel_eps],
                outputs=[self.control.joint_f],
                device=self.sim_device,
            )

            # step the arm dynamics only: particles are integrated by the VBD
            # solver below (one-way coupling, the arm does not feel the cloth)
            particle_count = self.model.particle_count
            self.model.particle_count = 0
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0.particle_f.zero_()
            self.model.particle_count = particle_count

            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    # --- task ------------------------------------------------------------------

    def _init_task(self, args):
        p = self.particles_per_world
        self.particle_q = wp.to_torch(self.state_0.particle_q).view(self.num_envs, p, 3)
        self.particle_qd = wp.to_torch(self.state_0.particle_qd).view(self.num_envs, p, 3)
        self.init_particle_q = self.particle_q.clone()

        self.cloth_start_xy = self.init_particle_q.mean(dim=1)[:, :2].clone()
        self.shoulder_xy = self.body_q[:, self.shoulder_body_local, :2].clone()
        self.goal_xy = self.cloth_start_xy.clone()
        self._cloth_goal_dist = torch.zeros(self.num_envs, device=self.device)

    def _reset_task(self, env_ids: torch.Tensor):
        self.particle_q[env_ids] = self.init_particle_q[env_ids]
        self.particle_qd[env_ids] = 0.0

        # goal on a ring around the cloth start, clamped into the reach disc
        n = len(env_ids)
        angle = 2.0 * torch.pi * torch.rand(n, device=self.device)
        radius = torch.empty(n, device=self.device).uniform_(*GOAL_RADIUS_RANGE)
        goal = self.cloth_start_xy[env_ids] + radius.unsqueeze(-1) * torch.stack(
            [torch.cos(angle), torch.sin(angle)], dim=-1
        )
        arm = goal - self.shoulder_xy[env_ids]
        dist = torch.linalg.norm(arm, dim=-1, keepdim=True).clamp(min=1e-6)
        goal = torch.where(
            dist > GOAL_MAX_FROM_SHOULDER, self.shoulder_xy[env_ids] + arm * (GOAL_MAX_FROM_SHOULDER / dist), goal
        )
        self.goal_xy[env_ids] = goal

    def cloth_center(self) -> torch.Tensor:
        return self.particle_q.mean(dim=1)

    def _compute_obs(self) -> torch.Tensor:
        cloth_center = self.cloth_center()
        goal_delta = self.goal_xy - cloth_center[:, :2]
        return torch.cat(
            [
                self.q - self.default_q,
                self.qd,
                cloth_center - self.ee_pos,
                torch.cat([goal_delta, torch.zeros_like(goal_delta[:, :1])], dim=-1),
                self.actions,
            ],
            dim=-1,
        )

    def _compute_rewards(self) -> torch.Tensor:
        cloth_center = self.cloth_center()
        d_reach = torch.linalg.norm(self.ee_pos - cloth_center, dim=-1)
        d_goal = torch.linalg.norm(cloth_center[:, :2] - self.goal_xy, dim=-1)
        self._cloth_goal_dist = d_goal

        # IsaacLab lift-style terms: reach the object, then track it to the goal
        reward = (
            1.0 * (1.0 - torch.tanh(d_reach / 0.1))
            + 16.0 * (1.0 - torch.tanh(d_goal / 0.15))
            + 5.0 * (1.0 - torch.tanh(d_goal / 0.05))
            - 1.0e-4 * torch.sum(torch.square(self.actions - self.prev_actions), dim=-1)
            - 1.0e-4 * torch.sum(torch.square(self.qd), dim=-1)
        )
        return reward * self.frame_dt

    def _log_dict(self) -> dict:
        return {"/metrics/cloth_goal_distance": self._cloth_goal_dist}


if __name__ == "__main__":
    parser = create_task_parser()
    force_null_viewer_for_training()
    viewer, args = newton.examples.init(parser)

    if args.num_envs is not None:
        num_envs = args.num_envs
    elif args.train:
        num_envs = 256
    elif args.test:
        num_envs = 2
    else:
        num_envs = 4

    env = SO101ClothEnv(args, num_envs)

    if args.train:
        train(env, args, TASK_NAME)
    else:
        policy = None
        try:
            policy = load_policy(env, args, TASK_NAME)
        except ImportError as exc:
            print(f"rsl_rl unavailable ({exc}); playing with a zero policy")
        newton.examples.run(ExamplePolicyPlayer(viewer, args, env, policy), args)
