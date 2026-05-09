"""
强化学习环境模块

基于 gymnasium 封装无人机避障环境，包含状态表示、动作执行、奖励计算和终止判断。
修正：添加地形碰撞检测，优化奖励计算，根据地形模式动态生成地形网格。
新增：支持最小高度和最大高度独立选择海拔/离地基准。
新增：在起点和终点附近（水平距离 ≤ goal_threshold）取消高度约束，允许起飞和降落。
新增：支持从配置传入 max_steps，控制每回合最大步数。
增强：step 返回的 info 字典包含更多字段用于轨迹分析和统计。
新增：全局-局部协同机制：A* 大步长路径 + 斜圆柱体约束域，状态空间扩展为12维。
奖励设计：基于 Potential-Based Shaping 势能重塑（单信号3D距离差分），
          + 飞行效率奖励（速度-目标方向对齐）+ 存活步奖励。
修正：移除距离信号三重计数（3D + XY + Z 分别差分），统一为 3D 势能差分。
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from data.models import MountainData, UAVConfig, AlgorithmConfig
from business.uav_builder import UAV
from business.obstacle_builder import check_collision_with_list
from business.terrain_builder import generate_terrain, generate_terrain_perlin
from business.global_path_planner import (
    GridMap, astar_planning, simplify_path, assign_altitude
)


class UAVEnv(gym.Env):
    """
    无人机避障强化学习环境（支持全局-局部协同）

    状态空间:
        - 无协同: 9 维 [x,y,z, vx,vy,vz, ax,ay,az]
        - 有协同: 12 维，增加 [dist_to_axis, progress, heading_error]
    动作空间: 3 维连续向量 [ax, ay, az] (加速度指令)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        mountain_data: MountainData,
        uav_config: UAVConfig,
        algorithm_config: AlgorithmConfig,
        dt: float = 0.1,
        goal_threshold: float = 2.0,
        max_steps: int = 500
    ):
        super().__init__()
        self.mountain_data = mountain_data
        self.uav_config = uav_config
        self.algorithm_config = algorithm_config
        self.dt = dt
        self.goal_threshold = goal_threshold
        self.max_steps = max_steps

        # 读取基准设置
        self.altitude_ref = uav_config.altitude_ref
        self.min_altitude_ref = uav_config.min_altitude_ref
        self.max_altitude_ref = uav_config.max_altitude_ref

        # 保存起点和终点的水平坐标
        self.start_xy = np.array([uav_config.start_x, uav_config.start_y])
        self.goal_xy = np.array([uav_config.goal_x, uav_config.goal_y])

        # 生成地形网格
        if mountain_data.terrain.terrain_mode == "perlin":
            self.X, self.Y, self.Z = generate_terrain_perlin(
                mountain_data.terrain,
                scale=mountain_data.terrain.perlin_scale,
                octaves=mountain_data.terrain.perlin_octaves,
                persistence=mountain_data.terrain.perlin_persistence,
                lacunarity=mountain_data.terrain.perlin_lacunarity,
                seed=mountain_data.terrain.perlin_seed,
                height_scale=mountain_data.terrain.perlin_height_scale
            )
        else:
            self.X, self.Y, self.Z = generate_terrain(mountain_data.terrain)

        self._build_terrain_height_map()

        # 起点和终点绝对高度
        start_terrain_h = self._get_terrain_height(uav_config.start_x, uav_config.start_y)
        goal_terrain_h = self._get_terrain_height(uav_config.goal_x, uav_config.goal_y)

        if self.altitude_ref == "agl":
            self.abs_start_z = start_terrain_h + uav_config.start_z
            self.abs_goal_z = goal_terrain_h + uav_config.goal_z
        else:
            self.abs_start_z = uav_config.start_z
            self.abs_goal_z = uav_config.goal_z

        # 创建 UAV 实例
        self.uav = UAV(uav_config)
        self.uav.position[2] = self.abs_start_z
        self.uav.goal[2] = self.abs_goal_z

        # ========== 全局-局部协同机制配置 ==========
        self.use_global_path = algorithm_config.use_global_path
        self.global_step_size = algorithm_config.global_step_size
        self.cylinder_radius = algorithm_config.cylinder_radius
        self.switch_threshold = algorithm_config.switch_threshold
        self.reward_path_follow = algorithm_config.reward_path_follow
        self.reward_path_progress = algorithm_config.reward_path_progress
        self.reward_out_of_cylinder = algorithm_config.reward_out_of_cylinder

        self._global_waypoints = None      # 3D 路径点 (N,3)
        self._current_seg_idx = 0
        self.current_start = None
        self.current_goal = None
        self.seg_vector = None
        self.seg_length = None
        self.seg_direction = None

        if self.use_global_path:
            self._compute_global_path()
            if self._global_waypoints is None:
                print("[UAVEnv] 警告：全局路径生成失败，将禁用协同机制")
                self.use_global_path = False

        # 状态空间边界
        x_min, x_max = mountain_data.terrain.x_min, mountain_data.terrain.x_max
        y_min, y_max = mountain_data.terrain.y_min, mountain_data.terrain.y_max
        z_min = uav_config.min_altitude if self.min_altitude_ref == "amsl" else -100
        z_max = uav_config.max_altitude if self.max_altitude_ref == "amsl" else 200
        v_max = uav_config.max_speed
        a_max = uav_config.max_accel

        if self.use_global_path:
            # 12 维状态空间
            low = np.array([
                x_min, y_min, z_min,
                -v_max, -v_max, -v_max,
                -a_max, -a_max, -a_max,
                0.0, 0.0, -1.0
            ])
            high = np.array([
                x_max, y_max, z_max,
                v_max, v_max, v_max,
                a_max, a_max, a_max,
                self.cylinder_radius, 1.0, 1.0
            ])
        else:
            # 9 维状态空间
            low = np.array([
                x_min, y_min, z_min,
                -v_max, -v_max, -v_max,
                -a_max, -a_max, -a_max
            ])
            high = np.array([
                x_max, y_max, z_max,
                v_max, v_max, v_max,
                a_max, a_max, a_max
            ])

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-a_max, high=a_max, shape=(3,), dtype=np.float32)

        self.steps = 0
        self.done = False

    # ---------- 全局路径生成 ----------
    def _compute_global_path(self):
        """使用 A* 生成全局路径并大步长简化"""
        # 提取障碍物（球体，忽略长方体简化处理）
        obstacles_xy = []
        for obs in self.mountain_data.obstacles:
            if hasattr(obs, 'radius'):
                obstacles_xy.append((obs.center_x, obs.center_y, obs.radius))
            elif hasattr(obs, 'length'):  # 长方体近似为球体（取半长轴）
                r = max(obs.length, obs.width) / 2
                obstacles_xy.append((obs.center_x, obs.center_y, r))

        # 创建栅格地图（分辨率 2 米）
        grid = GridMap(
            self.mountain_data.terrain.x_min,
            self.mountain_data.terrain.x_max,
            self.mountain_data.terrain.y_min,
            self.mountain_data.terrain.y_max,
            resolution=2.0,
            obstacles=obstacles_xy
        )

        start_xy = (self.uav_config.start_x, self.uav_config.start_y)
        goal_xy = (self.uav_config.goal_x, self.uav_config.goal_y)

        raw_path = astar_planning(grid, start_xy, goal_xy)
        if raw_path is None:
            print("[UAVEnv] A* 未找到可行路径")
            self.use_global_path = False
            return

        # 大步长简化
        simplified_xy = simplify_path(raw_path, self.global_step_size)
        # 赋予高度（安全高度 10 米）
        waypoints_3d = assign_altitude(simplified_xy, self._get_terrain_height, safe_height=10.0)
        self._global_waypoints = waypoints_3d
        print(f"[UAVEnv] 全局路径生成成功，路径段数: {len(waypoints_3d)-1}")

    def _update_current_segment(self):
        """更新当前路径段信息"""
        self.current_start = self._global_waypoints[self._current_seg_idx]
        self.current_goal = self._global_waypoints[self._current_seg_idx + 1]
        self.seg_vector = self.current_goal - self.current_start
        self.seg_length = np.linalg.norm(self.seg_vector)
        if self.seg_length > 0:
            self.seg_direction = self.seg_vector / self.seg_length
        else:
            self.seg_direction = np.zeros(3)

    def _get_path_features(self, pos):
        """
        计算与当前路径段相关的特征
        返回: (dist_to_axis, progress, heading_error)
        """
        if not self.use_global_path or self._global_waypoints is None:
            return 0.0, 0.0, 0.0

        # 投影到轴线上
        vec = pos - self.current_start
        t = np.dot(vec, self.seg_direction)
        t = np.clip(t, 0, self.seg_length)
        closest = self.current_start + t * self.seg_direction
        dist_to_axis = np.linalg.norm(pos - closest)
        progress = t / self.seg_length if self.seg_length > 0 else 0.0

        # 方向偏差（速度与指向当前路径点的夹角余弦）
        to_goal = self.current_goal - pos
        to_goal_norm = np.linalg.norm(to_goal)
        if to_goal_norm > 0:
            to_goal_dir = to_goal / to_goal_norm
        else:
            to_goal_dir = np.zeros(3)
        vel = self.uav.velocity
        vel_norm = np.linalg.norm(vel)
        if vel_norm > 0:
            heading_error = np.dot(vel / vel_norm, to_goal_dir)
        else:
            heading_error = 0.0

        return dist_to_axis, progress, heading_error

    # ---------- 辅助方法 ----------
    def _build_terrain_height_map(self):
        self.terrain_x = self.X[0, :]
        self.terrain_y = self.Y[:, 0]
        self.terrain_z = self.Z

    def _get_terrain_height(self, x, y):
        ix = np.argmin(np.abs(self.terrain_x - x))
        iy = np.argmin(np.abs(self.terrain_y - y))
        return self.terrain_z[iy, ix]

    def _check_terrain_collision(self):
        pos = self.uav.position
        terrain_h = self._get_terrain_height(pos[0], pos[1])
        return pos[2] < terrain_h

    def _check_out_of_bounds(self, pos):
        x_min, x_max = self.mountain_data.terrain.x_min, self.mountain_data.terrain.x_max
        y_min, y_max = self.mountain_data.terrain.y_min, self.mountain_data.terrain.y_max
        if pos[0] < x_min or pos[0] > x_max or pos[1] < y_min or pos[1] > y_max:
            return True

        # 高度范围检查（不考虑起点/终点附近）
        current_xy = pos[:2]
        dist_to_start = np.linalg.norm(current_xy - self.start_xy)
        dist_to_goal = np.linalg.norm(current_xy - self.goal_xy)
        near_start_or_goal = (dist_to_start <= self.goal_threshold) or (dist_to_goal <= self.goal_threshold)
        if not near_start_or_goal:
            terrain_h = self._get_terrain_height(pos[0], pos[1])
            if self.min_altitude_ref == "amsl":
                z_min_allowed = self.uav_config.min_altitude
            else:
                z_min_allowed = terrain_h + self.uav_config.min_altitude
            if self.max_altitude_ref == "amsl":
                z_max_allowed = self.uav_config.max_altitude
            else:
                z_max_allowed = terrain_h + self.uav_config.max_altitude
            if pos[2] < z_min_allowed or pos[2] > z_max_allowed:
                return True
        return False

    def _compute_reward(self, prev_pos, collision, out_of_bounds, dist_to_axis, progress, heading_error, reached_waypoint):
        """
        奖励计算（方案2：势能重塑版本）

        设计原则：
        1. Potential-Based Shaping：用距离势能 F(s) = -k·dist(s, goal) 驱动无人机靠近目标，
           γ·F(s') - F(s) ≈ k·(prev_dist - new_dist)，理论保证不改变最优策略，
           但消除稀疏奖励导致的局部最优陷阱。
        2. 飞行效率奖励：鼓励无人机朝目标方向飞行，速度方向与目标方向对齐时给正奖励。
        3. 平滑飞行罚项：抑制过大加速度，鼓励平稳飞行轨迹。
        4. 存活步奖励：每步小正奖励，防止策略主动触发碰撞结束回合。
        5. 碰撞/越界惩罚大额负分，路径跟随奖惩全局路径相关。
        """
        config = self.algorithm_config
        new_pos = self.uav.position

        # ---------- 势能函数：F(s) = -k * dist_to_goal ----------
        # Potential-Based Shaping 公式：r_shaped = r_env + γ·F(s') - F(s)
        # 简化（γ≈1）：r_shaped ≈ r_env + [F(s') - F(s)] = r_env + k·(prev_dist - new_dist)
        # 与原 reward_dist 类似，但物理意义更清晰，且支持更复杂的势能设计
        prev_dist_3d = np.linalg.norm(prev_pos - self.uav.goal)
        new_dist_3d = np.linalg.norm(new_pos - self.uav.goal)

        reward_potential = 0.0
        reward_efficiency = 0.0
        reward_smooth = 0.0
        reward_collision = 0.0
        reward_goal = 0.0
        reward_path = 0.0
        reward_progress = 0.0
        reward_out = 0.0
        reward_alive = 0.0  # 存活步奖励

        if collision or out_of_bounds:
            reward_collision = config.reward_collision
            # 无存活奖励，避免抵消碰撞惩罚
        else:
            # ===== 势能奖励（基于 Potential-Based Shaping）=====
            # 使用 3D 距离差分作为单一势能信号，避免距离信号三重计数。
            # F(s) = -k * dist_to_goal → γ*F(s') - F(s) ≈ k * (prev_dist - new_dist)
            # reward_xy + reward_z 被移除，因为 3D 距离差分已覆盖全方向。
            reward_potential = config.reward_dist * (prev_dist_3d - new_dist_3d)

            # ===== 飞行效率奖励：速度方向与目标方向对齐 =====
            # 鼓励无人机"向着目标飞"，而不是绕圈或横向漂移
            vel = self.uav.velocity
            vel_norm = np.linalg.norm(vel)
            if vel_norm > 0.5:  # 速度足够大才计算方向对齐（避免静止时噪声）
                to_goal = self.uav.goal - new_pos
                to_goal_norm = np.linalg.norm(to_goal)
                if to_goal_norm > self.goal_threshold:
                    cos_align = np.dot(vel / vel_norm, to_goal / to_goal_norm)
                    # cos_align ∈ [-1, 1]，只奖励正对齐（飞向目标）
                    reward_efficiency = 0.20 * max(0.0, cos_align)

            # ===== 平滑飞行奖励（保留原逻辑）=====
            reward_smooth = -config.reward_smooth * np.linalg.norm(self.uav.acceleration)

            # ===== 存活步奖励：每步+0.03，给予轻微正向激励防止策略主动求死，但不过度压倒目标奖励 =====
            reward_alive = 0.03

            # ===== 全局路径跟随奖励（保留原逻辑）=====
            if self.use_global_path:
                reward_path = self.reward_path_follow * dist_to_axis
                if progress > 0:
                    reward_progress = self.reward_path_progress * (progress - self._last_progress)
                self._last_progress = progress

        # 到达当前路径点奖励（保留原逻辑）
        if reached_waypoint:
            reward_goal += 20.0

        # 最终终点奖励（保留原逻辑）
        if new_dist_3d <= self.goal_threshold:
            reward_goal += 50.0

        # 超出圆柱体惩罚（保留原逻辑）
        if self.use_global_path and dist_to_axis > self.cylinder_radius:
            reward_out = self.reward_out_of_cylinder

        total_reward = (
            reward_potential + reward_efficiency +
            reward_smooth + reward_collision +
            reward_goal + reward_path + reward_progress +
            reward_out + reward_alive
        )

        components = {
            'dist': reward_potential,         # 势能奖励（含XY/Z分量）
            'efficiency': reward_efficiency,  # 飞行效率奖励（新增）
            'smooth': reward_smooth,
            'collision': reward_collision,
            'goal': reward_goal,
            'path_follow': reward_path,
            'progress': reward_progress,
            'out_of_cylinder': reward_out,
            'alive': reward_alive,            # 存活步奖励（新增）
        }
        return total_reward, components

    # ---------- 环境接口 ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.uav.reset()
        self.uav.position[2] = self.abs_start_z

        self.uav.goal[2] = self.abs_goal_z
        self.steps = 0
        self.done = False
        self._last_progress = 0.0

        if self.use_global_path and self._global_waypoints is not None:
            self._current_seg_idx = 0
            self._update_current_segment()

        state = self._get_state()
        return state.astype(np.float32), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        prev_pos = self.uav.position.copy()
        self.uav.set_acceleration(action)
        self.uav.update(self.dt)

        # 碰撞检测
        collision = check_collision_with_list(self.uav.position, self.mountain_data.obstacles)
        terrain_collision = self._check_terrain_collision()
        collision = collision or terrain_collision

        out_of_bounds = self._check_out_of_bounds(self.uav.position)

        # 全局路径相关
        dist_to_axis, progress, heading_error = self._get_path_features(self.uav.position)
        reached_waypoint = False
        if self.use_global_path and self._global_waypoints is not None:
            dist_to_waypoint = np.linalg.norm(self.uav.position - self.current_goal)
            if dist_to_waypoint <= self.switch_threshold and self._current_seg_idx < len(self._global_waypoints) - 2:
                self._current_seg_idx += 1
                self._update_current_segment()
                reached_waypoint = True

        # 奖励计算
        reward, components = self._compute_reward(
            prev_pos, collision, out_of_bounds,
            dist_to_axis, progress, heading_error, reached_waypoint
        )

        # 终止判断
        dist_to_final_goal = self.uav.distance_to_goal()
        terminated = (dist_to_final_goal <= self.goal_threshold) or collision or out_of_bounds
        self.steps += 1
        truncated = self.steps >= self.max_steps

        info = {
            "distance_to_goal": dist_to_final_goal,
            "steps": self.steps,
            "position": self.uav.position.copy(),
            "collision": collision,
            "terrain_collision": terrain_collision,
            "out_of_bounds": out_of_bounds,
            "min_altitude_ref": self.min_altitude_ref,
            "max_altitude_ref": self.max_altitude_ref,
            "near_start_or_goal": False,  # 简化，未使用
            "terrain_height": self._get_terrain_height(self.uav.position[0], self.uav.position[1]),
            "reached_goal": dist_to_final_goal <= self.goal_threshold,
            "reward_components": components,
            "dist_to_axis": dist_to_axis,
            "progress": progress,
            "heading_error": heading_error
        }

        state = self._get_state()
        return state.astype(np.float32), reward, terminated, truncated, info

    def _get_state(self):
        """返回当前状态（9维或12维）"""
        base = np.concatenate([self.uav.position, self.uav.velocity, self.uav.acceleration])
        if self.use_global_path and self._global_waypoints is not None:
            dist_to_axis, progress, heading_error = self._get_path_features(self.uav.position)
            return np.concatenate([base, [dist_to_axis, progress, heading_error]])
        else:
            return base

    def render(self, mode="human"):
        pass

    def close(self):
        pass