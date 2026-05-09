#!/usr/bin/env python3
"""
无头训练脚本 — 完全脱离 PyQt，纯 Python 训练循环。

预期提速 5~10 倍（vs UI 训练），模型和经验文件与 UI 完全互通，
训练后的模型可直接加载到 UI 中可视化回放。

用法：
    python scripts/train_headless.py --mountain "山区1" --config "无人机与算法_1"
    python scripts/train_headless.py --mountain "山区1" --config "无人机与算法_1" --epochs 5000 --log-interval 20

参数：
    --mountain      山区名称（对应桌面/算法训练集/{山区名}/）
    --config        配置名称（对应.../{山区名}/{配置名}/）
    --epochs        训练总轮数（默认从配置文件读取）
    --log-interval  日志打印间隔（默认 10 轮）
    --save-interval 模型保存间隔（默认 100 轮）
    --resume        是否加载已有检查点继续训练（默认 True）
"""

import sys
import os
import time
import argparse
import signal

# ── 将项目 src 目录加入 sys.path ──
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
from pathlib import Path
from datetime import datetime

from data.models import TrainJobConfig, TrainConfig
from data.serializer import load_json
from algorithm.uav_env import UAVEnv
from algorithm.ac_agent import ACAgent
from business.file_manager import FileManager
from business.checkpoint_manager import CheckpointManager


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="无头训练脚本（快速模式，无 GUI）")
    p.add_argument("--mountain", default="山区1", help="山区名称")
    p.add_argument("--config", default="无人机与算法_1", help="配置名称")
    p.add_argument("--epochs", type=int, default=None, help="训练轮数（覆盖配置文件）")
    p.add_argument("--log-interval", type=int, default=10, help="日志打印间隔（轮）")
    p.add_argument("--save-interval", type=int, default=100, help="模型保存间隔（轮）")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="从头开始，不加载已有检查点")
    p.add_argument("--envs", type=int, default=1,
                   help="并行环境数（>1 启用多 env 数据采集，默认 1）")
    p.set_defaults(resume=True)
    return p.parse_args()


def load_job_config(mountain_name: str, config_name: str) -> TrainJobConfig:
    """从 JSON 配置文件加载完整的训练任务配置"""
    fm = FileManager()
    config_path = fm.get_uav_alg_config_path(mountain_name, config_name)
    if not os.path.exists(config_path):
        print(f"[错误] 配置文件不存在: {config_path}")
        print(f"请确认山区 '{mountain_name}' 和配置 '{config_name}' 已保存。")
        sys.exit(1)
    data = load_json(config_path)
    return TrainJobConfig.from_dict(data)


def print_header(cfg: TrainJobConfig, args):
    """打印训练前信息总览"""
    sep = "=" * 60
    print()
    print(sep)
    print(f"  无头训练 — {args.mountain} / {args.config}")
    print(sep)
    print(f"  山区范围：         x=[{cfg.mountain.terrain.x_min}, {cfg.mountain.terrain.x_max}], "
          f"y=[{cfg.mountain.terrain.y_min}, {cfg.mountain.terrain.y_max}]")
    print(f"  起点 -> 终点：     ({cfg.uav.start_x}, {cfg.uav.start_y}, {cfg.uav.start_z}) "
          f"→ ({cfg.uav.goal_x}, {cfg.uav.goal_y}, {cfg.uav.goal_z})")
    print(f"  训练轮数：         {args.epochs or cfg.training.num_epochs}")
    print(f"  每轮最大步数：     {cfg.training.max_steps}")
    print(f"  经验池容量：       {cfg.algorithm.buffer_capacity}")
    print(f"  并行环境数：       {args.envs}")
    print(f"  日志间隔：         {args.log_interval} 轮")
    print(f"  模型保存间隔：     {args.save_interval} 轮")
    print(f"  奖励设定：")
    print(f"    - reward_dist         = {cfg.algorithm.reward_dist}")
    print(f"    - reward_collision    = {cfg.algorithm.reward_collision}")
    print(f"    - reward_smooth       = {cfg.algorithm.reward_smooth}")
    print(f"    - reward_path_follow  = {cfg.algorithm.reward_path_follow}")
    print(f"    - reward_path_progress= {cfg.algorithm.reward_path_progress}")
    print(f"    - reward_out_of_cyl   = {cfg.algorithm.reward_out_of_cylinder}")
    print(sep)
    print()


def main():
    args = parse_args()
    total_epochs = args.epochs

    # ── 加载配置 ──
    job_cfg = load_job_config(args.mountain, args.config)

    # 命令行覆盖训练轮数
    if total_epochs is None:
        total_epochs = job_cfg.training.num_epochs

    print_header(job_cfg, args)

    # ── 创建环境 ──
    max_steps = job_cfg.training.max_steps
    print(f"[初始化] 创建 UAVEnv...")
    env = UAVEnv(
        mountain_data=job_cfg.mountain,
        uav_config=job_cfg.uav,
        algorithm_config=job_cfg.algorithm,
        dt=0.1,
        goal_threshold=2.0,
        max_steps=max_steps,
    )
    state_dim = env.observation_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high
    print(f"[初始化] 状态空间: {state_dim} 维, 动作空间: 3 维")

    # ── 多环境并行支持 ──
    n_envs = args.envs
    envs = [env]
    if n_envs > 1:
        for i in range(1, n_envs):
            e = UAVEnv(
                mountain_data=job_cfg.mountain,
                uav_config=job_cfg.uav,
                algorithm_config=job_cfg.algorithm,
                dt=0.1,
                goal_threshold=2.0,
                max_steps=max_steps,
            )
            envs.append(e)
        print(f"[初始化] 并行环境: {n_envs} 个")

    # ── 创建智能体 ──
    agent = ACAgent(
        state_dim=state_dim,
        action_dim=3,
        action_low=action_low,
        action_high=action_high,
        lr_actor=job_cfg.algorithm.learning_rate,
        lr_critic=job_cfg.algorithm.learning_rate,
        gamma=job_cfg.algorithm.gamma,
        batch_size=job_cfg.algorithm.batch_size,
        hidden_size=job_cfg.algorithm.hidden_size,
        buffer_capacity=job_cfg.algorithm.buffer_capacity,
    )

    # ── 检查点管理 ──
    cm = CheckpointManager()
    cm.ensure_dirs(args.mountain, args.config)
    model_path = cm.get_model_path(args.mountain, args.config)
    current_epoch = 0

    # ── 加载已有检查点（续训） ──
    if args.resume:
        if cm.model_exists(args.mountain, args.config):
            agent.load(str(model_path), load_buffer=True)
            meta = cm.load_meta(args.mountain, args.config)
            if meta:
                current_epoch = meta.get("current_epoch", 0)
                print(f"[续训] 从轮次 {current_epoch}/{total_epochs} 恢复")
            else:
                print(f"[续训] 模型已加载，但元数据丢失，从 0 开始计数")
        else:
            print(f"[续训] 未找到已有模型，从头开始")

    # ── Ctrl+C 优雅退出 ──
    stop_requested = False

    def handle_sigint(sig, frame):
        nonlocal stop_requested
        if not stop_requested:
            print("\n[中断] 收到 Ctrl+C，正在保存模型... (再按一次强制退出)")
            stop_requested = True
        else:
            print("\n[中断] 强制退出")
            sys.exit(1)

    signal.signal(signal.SIGINT, handle_sigint)

    # ── 探索退火初始化 ──
    exploration_init = getattr(job_cfg.training, 'exploration_init', 1.5)

    # ── 训练主循环 ──
    print(f"\n[开始训练] {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'轮次':>8} {'成功':>5} {'碰撞':>5} {'步数':>6} {'奖励':>8} {'损失':>8} "
          f"{'dist':>8} {'耗时':>8} {'探索':>6}")
    print("-" * 70)

    train_start_time = time.time()
    epoch_times = []

    for epoch in range(current_epoch + 1, total_epochs + 1):
        if stop_requested:
            break

        epoch_start = time.time()
        progress = epoch / total_epochs

        # ── 探索退火 ──
        if hasattr(agent.actor, 'exploration_bonus'):
            bonus = max(0.0, exploration_init * (1.0 - progress))
            agent.set_exploration_bonus(bonus)

        # ── 多环境并行采集 ──
        all_obs = []
        all_infos = []
        for e in envs:
            obs, info = e.reset()
            all_obs.append(obs)
            all_infos.append(info)

        # 收集轨迹用于渲染（仅主 env）
        pos_trajectory = [all_obs[0][:3].copy()]

        total_reward = 0.0
        step_count = 0
        collisions = 0
        reached_goal = False
        loss_val = 0.0
        final_dist = 999.0

        dones = [False] * n_envs
        env_active = list(range(n_envs))

        # 每个 step 推进所有活跃 env，每步训练一次
        while env_active and not stop_requested:
            for idx in list(env_active):
                scaled_action, raw_action = agent.select_action(all_obs[idx])
                next_obs, reward, terminated, truncated, info = envs[idx].step(scaled_action)

                agent.store_transition(all_obs[idx], raw_action, reward,
                                       next_obs, terminated)

                all_obs[idx] = next_obs
                total_reward += reward / n_envs  # 取平均

                if info.get('collision', False):
                    collisions += 1

                done = terminated or truncated
                if done:
                    dones[idx] = True
                    env_active.remove(idx)
                    if info.get('reached_goal', False):
                        reached_goal = True
                    final_dist = info.get('distance_to_goal', final_dist)
                else:
                    # 记录主 env 轨迹
                    if idx == 0:
                        pos_trajectory.append(next_obs[:3].copy())

            step_count += 1

            # ── 每步训练（与原版 train_controller 行为一致）──
            train_info = agent.train()
            loss_val = train_info.get('actor_loss', loss_val)

            # 全 terminated 则退出循环
            if not env_active:
                break

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # ── 日志 ──
        if epoch % args.log_interval == 0 or epoch == 1:
            coll_rate = collisions / max(step_count, 1)
            elapsed = time.time() - train_start_time
            eta_seconds = (total_epochs - epoch) * np.mean(epoch_times[-100:]) if epoch_times else 0
            eta_str = f"{eta_seconds / 60:.0f}m" if eta_seconds > 60 else f"{eta_seconds:.0f}s"

            print(f"{epoch:>8d} {'Y' if reached_goal else 'N':>5} "
                  f"{coll_rate:>5.2f} {step_count:>6d} "
                  f"{total_reward:>8.1f} {loss_val:>8.4f} "
                  f"{final_dist:>8.2f} {epoch_time:>7.2f}s "
                  f"{bonus:>6.2f}",
                  flush=True)

        # ── 保存模型 & 元数据 ──
        if epoch % args.save_interval == 0 or epoch == 1:
            agent.save(str(model_path), save_buffer=True)
            cm.save_meta(
                args.mountain, args.config,
                total_epochs=total_epochs,
                current_epoch=epoch,
                additional_info={
                    "save_interval": args.save_interval,
                    "headless": True,
                }
            )

            # 保存单轮统计
            stats = {
                'episode': epoch,
                'success': reached_goal,
                'collision': collisions > 0,
                'final_distance': float(final_dist),
                'total_reward': float(total_reward),
                'steps': step_count,
                'loss': float(loss_val),
                'exploration_bonus': float(bonus),
            }
            cm.append_stats(args.mountain, args.config, stats)
            print(f"  → 检查点已保存 (轮次 {epoch})")

    # ── 训练结束 ──
    total_time = time.time() - train_start_time
    print("-" * 70)
    print(f"[完成] 总耗时: {total_time / 60:.1f} 分钟 "
          f"({total_time:.0f} 秒)")
    print(f"[完成] 平均每轮: {np.mean(epoch_times):.2f}s "
          f"(共 {len(epoch_times)} 轮)")

    # 最终保存
    if not stop_requested or True:  # 中断时也保存
        agent.save(str(model_path), save_buffer=True)
        cm.save_meta(
            args.mountain, args.config,
            total_epochs=total_epochs,
            current_epoch=epoch,
        )
        print(f"[完成] 最终模型已保存: {model_path}")

    # 清理
    for e in envs:
        e.close()

    print("[完成] 训练结束，模型可加载至 UI 进行可视化回放")


if __name__ == "__main__":
    main()
