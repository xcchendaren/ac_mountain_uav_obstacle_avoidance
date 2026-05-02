"""
训练流程控制模块

封装训练启动、暂停、数据保存等控制逻辑，集成真实 UAVEnv 和 ACAgent。
修正：递归调度、适配新动作接口、模型保存路径独立。
新增：保存扩展轨迹（13维），收集每轮统计指标（含奖励成分分析）。
新增：模型和缓冲区保存到配置文件夹（与训练过程文件夹并列），实现跨训练过程经验继承。
新增：支持自定义模型保存路径，允许用户在训练前指定模型和经验文件的保存位置。
新增：save_experience_only 方法，仅保存经验缓冲区。
新增：自动创建空白经验文件，确保训练开始前经验文件存在。
新增：持续训练模式，训练直到成功到达终点后自动停止。
优化：使用 save_dir 统一管理保存目录，支持训练中动态切换保存位置。
新增：每100轮更新经验池奖励阈值（调用 agent.update_buffer_threshold()）。
新增：集成全局-局部协同机制，从 AlgorithmConfig 读取参数并传入环境。
"""

import numpy as np
import os
import re
import pickle
from collections import deque
from pathlib import Path
from PyQt5.QtCore import QObject, pyqtSignal, QTimer

from business.file_manager import FileManager
from data.serializer import save_npy
from algorithm.ac_agent import ACAgent
from algorithm.uav_env import UAVEnv
from data.models import TrainJobConfig


class TrainController(QObject):
    """
    训练流程控制器（真实训练版本）
    """

    training_started = pyqtSignal()
    training_paused = pyqtSignal()
    training_stopped = pyqtSignal()
    epoch_completed = pyqtSignal(int, dict)  # 轮次，统计信息字典（含终止原因和奖励成分）
    trajectory_generated = pyqtSignal(np.ndarray)  # 轨迹数据 (n, 3) 位置信息，用于实时显示
    progress_updated = pyqtSignal(float)  # 训练进度 (0~1)

    def __init__(self, main_window, file_manager: FileManager, job_config: TrainJobConfig,
                 mountain_name="山区1", config_name="无人机与算法_1", run_index=1,
                 save_dir=None):
        """
        初始化训练控制器

        Args:
            main_window: 主窗口引用
            file_manager: 文件管理器
            job_config: 训练任务配置
            mountain_name: 山区名称
            config_name: 配置名称
            run_index: 本次训练的序号（第几次训练过程）
            save_dir: 保存目录（模型和经验文件的存储位置），若为None则使用默认配置文件夹
        """
        super().__init__()
        self.main_window = main_window
        self.file_manager = file_manager
        self.job_config = job_config
        self.mountain_name = mountain_name
        self.config_name = config_name
        self.run_index = run_index
        self.save_dir = Path(save_dir).resolve() if save_dir else None

        self.is_training = False
        self.is_paused = False
        self.current_epoch = 0
        self.total_epochs = job_config.training.num_epochs
        self.save_interval = job_config.training.save_interval

        self.env = None
        self.agent = None
        self.run_dir = None
        self.config_dir = None
        self.model_path = None  # 模型保存的完整路径

        # 持续训练模式标志
        self.continuous_mode = False

        # 用于递归调度的标志
        self._pending_call = False

        self.log("训练控制器初始化完成")

    def log(self, msg):
        self.main_window.log(f"[TrainController] {msg}")

    def start_training(self):
        if self.is_training:
            self.log("训练已在运行中")
            return

        # 1. 构建全局-局部协同配置字典
        alg_cfg = self.job_config.algorithm
        global_planner_config = {
            'enabled': getattr(alg_cfg, 'use_global_path', False),
            'step_size': getattr(alg_cfg, 'global_step_size', 15.0),
            'cylinder_radius': getattr(alg_cfg, 'cylinder_radius', 20.0),
            'switch_threshold': getattr(alg_cfg, 'switch_threshold', 5.0),
            'reward_path_follow': getattr(alg_cfg, 'reward_path_follow', -0.5),
            'reward_path_progress': getattr(alg_cfg, 'reward_path_progress', 1.0),
            'reward_out_of_cylinder': getattr(alg_cfg, 'reward_out_of_cylinder', -5.0),
        }

        # 2. 创建环境（传入协同配置）
        self.env = UAVEnv(
            mountain_data=self.job_config.mountain,
            uav_config=self.job_config.uav,
            algorithm_config=self.job_config.algorithm,
            dt=0.1,
            goal_threshold=2.0,
            max_steps=self.job_config.training.max_steps,
        )
        state_dim = self.env.observation_space.shape[0]   # 自动适应 9 或 12
        action_low = self.env.action_space.low
        action_high = self.env.action_space.high

        # 3. 创建智能体
        self.agent = ACAgent(
            state_dim=state_dim,
            action_dim=3,
            action_low=action_low,
            action_high=action_high,
            lr_actor=self.job_config.algorithm.learning_rate,
            lr_critic=self.job_config.algorithm.learning_rate,
            gamma=self.job_config.algorithm.gamma,
            batch_size=self.job_config.algorithm.batch_size,
            hidden_size=self.job_config.algorithm.hidden_size,
            buffer_capacity=self.job_config.algorithm.buffer_capacity,
        )

        # 4. 确定模型保存路径
        if self.save_dir:
            self.model_path = self.save_dir / "model.pth"
            self.config_dir = self.save_dir
        else:
            config_path_str = self.file_manager.get_uav_alg_dir_path(self.mountain_name, self.config_name)
            self.config_dir = Path(config_path_str).resolve()
            self.model_path = self.config_dir / "model.pth"

        self.log(f"模型保存路径: {self.model_path}")
        self.log(f"配置文件夹路径: {self.config_dir}")
        # 可视化全局路径和圆柱体
        if self.env.use_global_path and self.env._global_waypoints is not None:
            self.main_window.renderer.draw_global_path(self.env._global_waypoints)
            self.main_window.renderer.draw_cylinder_segments(
                self.env._global_waypoints,
                self.job_config.algorithm.cylinder_radius
            )
            self.log("已绘制全局路径及圆柱体约束域")

        # 5. 确保目录存在
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.log(f"创建配置目录失败: {e}")
            return

        # 6. 确保经验文件存在（如果模型文件不存在，但缓冲区文件可能不存在，则创建空白缓冲区）
        buffer_path = self.model_path.with_suffix('.buffer.pkl')
        if not self.model_path.exists() and not buffer_path.exists():
            try:
                empty_buffer = deque(maxlen=self.job_config.algorithm.buffer_capacity)
                with open(buffer_path, 'wb') as f:
                    pickle.dump(empty_buffer, f)
                self.log(f"创建空白经验文件: {buffer_path}")
            except Exception as e:
                self.log(f"创建空白经验文件失败: {e}，将使用内存中的空缓冲区")

        # 7. 尝试加载已有模型和缓冲区
        if self.model_path.exists():
            try:
                self.agent.load(str(self.model_path), load_buffer=True)
                self.log("已加载历史模型和缓冲区")
            except Exception as e:
                self.log(f"加载历史模型失败: {e}，将从头开始训练")
        else:
            self.log("未找到历史模型，从头开始训练")
            if buffer_path.exists():
                try:
                    with open(buffer_path, 'rb') as f:
                        loaded_buffer = pickle.load(f)
                        self.agent.replay_buffer.buffer = loaded_buffer
                        self.agent.replay_buffer.rewards = [exp[2] for exp in loaded_buffer]
                        self.agent.replay_buffer.threshold = -np.inf
                        self.agent.replay_buffer.need_update = True
                    self.log(f"已单独加载经验缓冲区: {buffer_path}")
                except Exception as e:
                    self.log(f"加载经验缓冲区失败: {e}")

        # 8. 创建训练过程目录（轨迹保存位置）
        self.run_dir = Path(
            self.file_manager.create_train_dirs(
                self.mountain_name, self.config_name, self.run_index
            )
        ).resolve()
        self.log(f"训练数据将保存至: {self.run_dir}")

        # 9. 从 run_dir 中推断已完成的轮次
        max_epoch = 0
        pattern = re.compile(r'第(\d+)轮训练过程\.npy')
        for f in self.run_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                epoch_num = int(m.group(1))
                max_epoch = max(max_epoch, epoch_num)
        self.current_epoch = max_epoch
        if self.current_epoch > 0:
            self.log(f"从检查点恢复，当前进度: {self.current_epoch}/{self.total_epochs}")
        else:
            self.current_epoch = 0

        # 10. 启动训练
        self.is_training = True
        self.is_paused = False
        self.training_started.emit()
        self.progress_updated.emit(0.0)

        # 11. 立即开始第一轮
        self._schedule_next()

    def _schedule_next(self):
        """调度下一轮训练（立即执行）"""
        if not self.is_training or self.is_paused or self._pending_call:
            return
        self._pending_call = True
        QTimer.singleShot(0, self._run_one_epoch)

    def _run_one_epoch(self):
        self._pending_call = False
        if not self.is_training or self.is_paused:
            return

        # 持续训练模式下，不检查轮次上限
        if not self.continuous_mode and self.current_epoch >= self.total_epochs:
            self.stop_training()
            return

        epoch = self.current_epoch + 1
        obs, info = self.env.reset()
        # 用于实时显示的位置轨迹（仅位置）
        pos_trajectory = [obs[:3].copy()]
        # 用于保存的扩展轨迹（每步13维）
        extended_trajectory = []
        total_reward = 0
        done = False
        step_count = 0

        # 统计相关变量
        acc_list = []                 # 记录每一步的加速度大小
        collision_occurred = False    # 是否发生碰撞
        path_length = 0.0              # 累计路径长度
        prev_pos = obs[:3].copy()     # 上一步位置（用于计算路径长度）

        # 奖励成分累加器
        total_reward_dist = 0.0
        total_reward_smooth = 0.0
        total_reward_collision = 0.0
        total_reward_goal = 0.0

        while not done:
            # 选择动作
            scaled_action, raw_action = self.agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(scaled_action)
            # 存储经验
            self.agent.store_transition(obs, raw_action, reward, next_obs, terminated)
            train_info = self.agent.train()  # 训练返回的损失等信息

            # 收集扩展数据
            ground_clearance = next_obs[2] - info.get('terrain_height', 0)
            collision_flag = 1.0 if info.get('collision', False) else 0.0
            dist_to_goal = info.get('distance_to_goal', 0.0)
            curr_pos = next_obs[:3]
            step_distance = np.linalg.norm(curr_pos - prev_pos)
            path_length += step_distance
            prev_pos = curr_pos

            acc = next_obs[6:9]
            acc_mag = np.linalg.norm(acc)
            acc_list.append(acc_mag)

            if info.get('collision', False):
                collision_occurred = True

            components = info.get('reward_components', {})
            total_reward_dist += components.get('dist', 0.0)
            total_reward_smooth += components.get('smooth', 0.0)
            total_reward_collision += components.get('collision', 0.0)
            total_reward_goal += components.get('goal', 0.0)

            step_data = np.concatenate([
                next_obs[:3],
                next_obs[3:6],
                next_obs[6:9],
                [ground_clearance],
                [collision_flag],
                [dist_to_goal],
                [path_length]
            ])
            extended_trajectory.append(step_data)

            pos_trajectory.append(curr_pos.copy())
            total_reward += reward
            obs = next_obs
            done = terminated or truncated
            step_count += 1

        extended_trajectory = np.array(extended_trajectory)
        pos_trajectory = np.array(pos_trajectory)

        # 保存扩展轨迹（按间隔）
        if epoch % self.save_interval == 0 or epoch == 1:
            save_path = self.file_manager.get_trajectory_path(
                self.mountain_name, self.config_name, self.run_index, epoch)
            save_npy(extended_trajectory, save_path)
            self.log(f"已保存第{epoch}轮扩展轨迹 (13维)")

        # 计算统计指标
        success = (dist_to_goal <= self.env.goal_threshold) if hasattr(self.env, 'goal_threshold') else False
        avg_acc = np.mean(acc_list) if acc_list else 0.0
        acc_var = np.var(acc_list) if acc_list else 0.0
        loss_value = train_info.get('actor_loss', 0.0)

        if info.get('reached_goal', False):
            reason = "到达终点"
        elif info.get('collision', False):
            reason = "碰撞"
        elif info.get('out_of_bounds', False):
            reason = "越界"
        elif info.get('terrain_collision', False):
            reason = "地形碰撞"
        elif step_count >= self.job_config.training.max_steps:
            reason = f"步数超限({step_count})"
        else:
            reason = "未知"

        stats = {
            'episode': epoch,
            'success': success,
            'collision': collision_occurred,
            'final_distance': dist_to_goal,
            'path_length': path_length,
            'avg_acceleration': avg_acc,
            'acceleration_variance': acc_var,
            'total_reward': total_reward,
            'steps': step_count,
            'loss': loss_value,
            'termination_reason': reason,
            'reward_dist': total_reward_dist,
            'reward_smooth': total_reward_smooth,
            'reward_collision': total_reward_collision,
            'reward_goal': total_reward_goal,
        }

        self.epoch_completed.emit(epoch, stats)
        self.trajectory_generated.emit(pos_trajectory)

        self.current_epoch += 1
        progress = self.current_epoch / self.total_epochs
        self.progress_updated.emit(progress)

        # 每隔100轮更新经验池奖励阈值
        if epoch % 100 == 0:
            self.agent.update_buffer_threshold()
            self.log(f"已更新经验池奖励阈值 (第{epoch}轮)")

        # 每隔 save_interval 轮保存模型
        if epoch % self.save_interval == 0 or epoch == 1:
            self.save_checkpoint()

        # 持续训练模式：如果本轮成功到达终点，则停止训练
        if self.continuous_mode and success:
            self.log(f"第{epoch}轮成功到达终点，持续训练完成")
            self.stop_training()
            return

        self._schedule_next()

    def pause_training(self):
        if self.is_training and not self.is_paused:
            self.is_paused = True
            self.log("训练已暂停")
            self.training_paused.emit()

    def resume_training(self):
        if self.is_training and self.is_paused:
            self.is_paused = False
            self.log("训练恢复")
            self._schedule_next()

    def stop_training(self):
        if self.is_training:
            self.save_checkpoint()
            self.is_training = False
            self.is_paused = False
            self._pending_call = False
            self.continuous_mode = False
            self.log("训练已停止")
            self.training_stopped.emit()
            if self.env is not None:
                self.env.close()
                self.env = None
            self.agent = None

    def save_checkpoint(self):
        """保存模型和经验缓冲区到指定路径"""
        if self.agent is not None and self.model_path is not None:
            try:
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                self.agent.save(str(self.model_path), save_buffer=True)
                self.log(f"模型检查点已保存至: {self.model_path}")
            except Exception as e:
                self.log(f"保存模型失败: {e}")
                import traceback
                traceback.print_exc()

    def save_experience_only(self):
        """仅保存经验缓冲区（不保存模型）"""
        if self.agent is None or self.model_path is None:
            self.log("无法保存经验：智能体或模型路径不存在")
            return
        try:
            buffer_path = self.model_path.with_suffix('.buffer.pkl')
            buffer_path.parent.mkdir(parents=True, exist_ok=True)
            with open(buffer_path, 'wb') as f:
                pickle.dump(self.agent.replay_buffer.buffer, f)
            self.log(f"经验缓冲区已保存至: {buffer_path}")
        except Exception as e:
            self.log(f"保存经验缓冲区失败: {e}")
            import traceback
            traceback.print_exc()

    def load_checkpoint(self, model_path):
        if self.agent is not None and Path(model_path).exists():
            self.agent.load(model_path, load_buffer=True)
            self.log(f"模型已从 {model_path} 加载")

    def get_progress(self):
        if self.total_epochs == 0:
            return 0
        return self.current_epoch / self.total_epochs

    def change_save_dir(self, new_dir, save_current=True):
        """
        动态更改保存目录
        :param new_dir: 新目录路径（Path或str）
        :param save_current: 是否将当前经验缓冲区保存到新目录
        """
        new_dir = Path(new_dir).resolve()
        if self.is_training:
            if save_current and self.agent is not None:
                buffer_path = new_dir / "model.buffer.pkl"
                buffer_path.parent.mkdir(parents=True, exist_ok=True)
                with open(buffer_path, 'wb') as f:
                    pickle.dump(self.agent.replay_buffer.buffer, f)
                self.log(f"[TrainController] 已将当前经验保存到新目录: {buffer_path}")

        self.save_dir = new_dir
        self.model_path = self.save_dir / "model.pth"
        self.config_dir = self.save_dir
        self.log(f"[TrainController] 保存目录已更改为: {self.save_dir}")