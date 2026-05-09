"""
训练检查点管理模块

统一管理训练过程中的模型、经验缓冲区和轨迹保存。
支持中文路径，使用 pathlib.Path 处理路径确保兼容性。

目录结构：
    {root_dir}/
    └── {mountain_name}/
        └── {config_name}/
            ├── checkpoint/
            │   ├── model.pth           # 神经网络模型
            │   ├── buffer.pkl          # 经验缓冲区
            │   └── meta.json           # 元数据（时间、轮次等）
            └── trajectories/
                ├── episode_000010.npy  # 单轮扩展轨迹
                └── stats.json          # 统计信息汇总
"""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import numpy as np


def _convert_to_native_types(obj: Any) -> Any:
    """
    递归转换 numpy 类型为 Python 原生类型，确保 JSON 序列化兼容

    Args:
        obj: 任意对象（可能是 numpy 数组、numpy 标量等）

    Returns:
        Python 原生类型对象
    """
    if isinstance(obj, np.ndarray):
        # numpy 数组转换为列表
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        # numpy 标量转换为 Python 原生类型
        return obj.item()
    elif isinstance(obj, np.bool_):
        # numpy bool_ 转换为 Python bool
        return bool(obj)
    elif isinstance(obj, dict):
        # 递归处理字典
        return {k: _convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # 递归处理列表/元组
        return [_convert_to_native_types(item) for item in obj]
    else:
        # 其他类型保持不变
        return obj


class CheckpointManager:
    """
    训练检查点管理器

    统一管理模型、经验缓冲区和轨迹的保存与加载。
    """

    # 子目录名称
    CHECKPOINT_DIR = "checkpoint"
    TRAJECTORIES_DIR = "trajectories"

    # 文件名称
    MODEL_FILE = "model.pth"
    BUFFER_FILE = "buffer.pkl"
    META_FILE = "meta.json"
    STATS_FILE = "stats.json"

    def __init__(self, root_dir: Optional[str] = None):
        """
        初始化检查点管理器

        Args:
            root_dir: 训练集根目录，若为None则使用默认路径
        """
        if root_dir is None:
            desktop = Path.home() / "Desktop"
            self.root_dir = desktop / "算法训练集"
        else:
            self.root_dir = Path(root_dir)

        self.root_dir.mkdir(parents=True, exist_ok=True)
        print(f"[CheckpointManager] 初始化，根目录: {self.root_dir}")

    # =========================================================
    # 路径生成
    # =========================================================

    def get_base_path(self, mountain_name: str, config_name: str) -> Path:
        """
        获取配置基础路径

        Args:
            mountain_name: 山区名称
            config_name: 配置名称

        Returns:
            Path: {root_dir}/{mountain_name}/{config_name}/
        """
        return self.root_dir / mountain_name / config_name

    def get_checkpoint_dir(self, mountain_name: str, config_name: str) -> Path:
        """获取检查点目录路径"""
        return self.get_base_path(mountain_name, config_name) / self.CHECKPOINT_DIR

    def get_trajectories_dir(self, mountain_name: str, config_name: str) -> Path:
        """获取轨迹目录路径"""
        return self.get_base_path(mountain_name, config_name) / self.TRAJECTORIES_DIR

    def get_model_path(self, mountain_name: str, config_name: str) -> Path:
        """获取模型文件路径"""
        return self.get_checkpoint_dir(mountain_name, config_name) / self.MODEL_FILE

    def get_buffer_path(self, mountain_name: str, config_name: str) -> Path:
        """获取缓冲区文件路径"""
        return self.get_checkpoint_dir(mountain_name, config_name) / self.BUFFER_FILE

    def get_meta_path(self, mountain_name: str, config_name: str) -> Path:
        """获取元数据文件路径"""
        return self.get_checkpoint_dir(mountain_name, config_name) / self.META_FILE

    def get_trajectory_path(self, mountain_name: str, config_name: str, episode: int) -> Path:
        """
        获取单轮轨迹文件路径

        Args:
            mountain_name: 山区名称
            config_name: 配置名称
            episode: 轮次编号

        Returns:
            Path: trajectories/episode_000010.npy
        """
        filename = f"episode_{episode:06d}.npy"
        return self.get_trajectories_dir(mountain_name, config_name) / filename

    def get_stats_path(self, mountain_name: str, config_name: str) -> Path:
        """获取统计汇总文件路径"""
        return self.get_trajectories_dir(mountain_name, config_name) / self.STATS_FILE

    # =========================================================
    # 目录创建
    # =========================================================

    def ensure_dirs(self, mountain_name: str, config_name: str) -> None:
        """
        确保所有必要的目录存在

        Args:
            mountain_name: 山区名称
            config_name: 配置名称
        """
        checkpoint_dir = self.get_checkpoint_dir(mountain_name, config_name)
        trajectories_dir = self.get_trajectories_dir(mountain_name, config_name)

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trajectories_dir.mkdir(parents=True, exist_ok=True)

        print(f"[CheckpointManager] 目录已创建: {checkpoint_dir}")
        print(f"[CheckpointManager] 目录已创建: {trajectories_dir}")

    # =========================================================
    # 元数据管理
    # =========================================================

    def load_meta(self, mountain_name: str, config_name: str) -> Optional[Dict[str, Any]]:
        """
        加载元数据

        Returns:
            Dict: 元数据字典，若不存在则返回None
        """
        meta_path = self.get_meta_path(mountain_name, config_name)
        if not meta_path.exists():
            return None

        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[CheckpointManager] 加载元数据失败: {e}")
            return None

    def save_meta(self, mountain_name: str, config_name: str,
                  total_epochs: int, current_epoch: int,
                  additional_info: Optional[Dict[str, Any]] = None) -> None:
        """
        保存元数据

        Args:
            mountain_name: 山区名称
            config_name: 配置名称
            total_epochs: 总训练轮数
            current_epoch: 当前训练轮数
            additional_info: 额外信息
        """
        meta_path = self.get_meta_path(mountain_name, config_name)

        # 确保目录存在
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "mountain_name": mountain_name,
            "config_name": config_name,
            "total_epochs": total_epochs,
            "current_epoch": current_epoch,
            "last_updated": datetime.now().isoformat(),
        }

        if additional_info:
            meta.update(additional_info)

        # 转换 numpy 类型
        meta = _convert_to_native_types(meta)

        try:
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[CheckpointManager] 保存元数据失败: {e}")

    # =========================================================
    # 统计信息管理
    # =========================================================

    def load_stats(self, mountain_name: str, config_name: str) -> List[Dict[str, Any]]:
        """
        加载统计信息列表

        Returns:
            List[Dict]: 统计信息列表
        """
        stats_path = self.get_stats_path(mountain_name, config_name)
        if not stats_path.exists():
            return []

        try:
            with open(stats_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 验证数据格式
                if isinstance(data, list):
                    return data
                else:
                    print(f"[CheckpointManager] 统计信息格式错误（期望列表），已重置")
                    return []
        except (json.JSONDecodeError, IOError) as e:
            print(f"[CheckpointManager] 加载统计信息失败: {e}")
            # 文件可能损坏，尝试备份并返回空列表
            try:
                backup_path = stats_path.with_suffix('.json.bak')
                stats_path.rename(backup_path)
                print(f"[CheckpointManager] 已备份损坏的文件到: {backup_path}")
            except Exception:
                pass
            return []

    def append_stats(self, mountain_name: str, config_name: str,
                     stats: Dict[str, Any]) -> None:
        """
        追加单轮统计信息

        Args:
            mountain_name: 山区名称
            config_name: 配置名称
            stats: 单轮统计信息字典
        """
        stats_path = self.get_stats_path(mountain_name, config_name)

        # 加载现有统计
        all_stats = self.load_stats(mountain_name, config_name)

        # 转换统计信息中的 numpy 类型为 Python 原生类型
        stats_converted = _convert_to_native_types(stats)

        # 追加新统计
        all_stats.append(stats_converted)

        # 保存前确保目录存在
        stats_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存
        try:
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(all_stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[CheckpointManager] 保存统计信息失败: {e}")

    # =========================================================
    # 模型和缓冲区管理
    # =========================================================

    def model_exists(self, mountain_name: str, config_name: str) -> bool:
        """检查模型文件是否存在"""
        return self.get_model_path(mountain_name, config_name).exists()

    def buffer_exists(self, mountain_name: str, config_name: str) -> bool:
        """检查缓冲区文件是否存在"""
        return self.get_buffer_path(mountain_name, config_name).exists()

    def save_buffer(self, mountain_name: str, config_name: str,
                    buffer_data: Any) -> bool:
        """
        保存经验缓冲区

        Args:
            mountain_name: 山区名称
            config_name: 配置名称
            buffer_data: 缓冲区数据

        Returns:
            bool: 是否保存成功
        """
        buffer_path = self.get_buffer_path(mountain_name, config_name)

        # 确保目录存在
        buffer_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(buffer_path, 'wb') as f:
                pickle.dump(buffer_data, f)
            print(f"[CheckpointManager] 缓冲区已保存: {buffer_path}")
            return True
        except Exception as e:
            print(f"[CheckpointManager] 保存缓冲区失败: {e}")
            return False

    def load_buffer(self, mountain_name: str, config_name: str) -> Optional[Any]:
        """
        加载经验缓冲区

        Returns:
            Any: 缓冲区数据，失败返回None
        """
        buffer_path = self.get_buffer_path(mountain_name, config_name)
        if not buffer_path.exists():
            return None

        try:
            with open(buffer_path, 'rb') as f:
                data = pickle.load(f)
            print(f"[CheckpointManager] 缓冲区已加载: {buffer_path}")
            return data
        except Exception as e:
            print(f"[CheckpointManager] 加载缓冲区失败: {e}")
            return None

    # =========================================================
    # 轨迹管理
    # =========================================================

    def get_saved_episodes(self, mountain_name: str, config_name: str) -> List[int]:
        """
        获取已保存的轨迹轮次列表

        Returns:
            List[int]: 轮次编号列表
        """
        traj_dir = self.get_trajectories_dir(mountain_name, config_name)
        if not traj_dir.exists():
            return []

        episodes = []
        prefix = "episode_"
        suffix = ".npy"

        for f in traj_dir.iterdir():
            if f.suffix == suffix and f.stem.startswith(prefix):
                try:
                    episode_num = int(f.stem[len(prefix):])
                    episodes.append(episode_num)
                except ValueError:
                    continue

        return sorted(episodes)

    def get_latest_checkpoint_info(self, mountain_name: str, config_name: str) -> Optional[Dict[str, Any]]:
        """
        获取最新检查点信息

        Returns:
            Dict: 包含current_epoch等信息，若无检查点返回None
        """
        meta = self.load_meta(mountain_name, config_name)
        if meta is None:
            return None

        return {
            "current_epoch": meta.get("current_epoch", 0),
            "total_epochs": meta.get("total_epochs", 0),
            "last_updated": meta.get("last_updated", ""),
        }

    # =========================================================
    # 工具方法
    # =========================================================

    def get_all_configs(self) -> List[Dict[str, str]]:
        """
        获取所有配置信息列表

        Returns:
            List[Dict]: [{mountain_name, config_name}, ...]
        """
        configs = []

        if not self.root_dir.exists():
            return configs

        for mountain_dir in self.root_dir.iterdir():
            if mountain_dir.is_dir():
                for config_dir in mountain_dir.iterdir():
                    if config_dir.is_dir():
                        checkpoint = config_dir / self.CHECKPOINT_DIR
                        if checkpoint.exists() and (checkpoint / self.MODEL_FILE).exists():
                            configs.append({
                                "mountain_name": mountain_dir.name,
                                "config_name": config_dir.name,
                                "base_path": str(config_dir),
                            })

        return configs


if __name__ == "__main__":
    # 测试
    cm = CheckpointManager()

    # 测试路径生成
    mountain = "测试山区"
    config = "测试配置"

    print(f"基础路径: {cm.get_base_path(mountain, config)}")
    print(f"检查点目录: {cm.get_checkpoint_dir(mountain, config)}")
    print(f"轨迹目录: {cm.get_trajectories_dir(mountain, config)}")
    print(f"模型路径: {cm.get_model_path(mountain, config)}")
    print(f"第10轮轨迹: {cm.get_trajectory_path(mountain, config, 10)}")

    # 测试目录创建
    cm.ensure_dirs(mountain, config)

    # 测试元数据
    cm.save_meta(mountain, config, total_epochs=1000, current_epoch=100)
    meta = cm.load_meta(mountain, config)
    print(f"元数据: {meta}")
