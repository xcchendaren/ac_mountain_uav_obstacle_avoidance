"""
数据模型定义模块

包含山区地形、障碍物、无人机配置、算法配置等数据结构。
所有模型均设计为可序列化为字典/JSON的格式。
"""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Union
import numpy as np


@dataclass
class TerrainData:
    """
    山区地形建模参数

    基于函数建模法：z(x,y) = sin(y+a) + b·sin(x) + c·cos(d·√(x²+y²)) + e·cos(y) + f·sin(g·√(x²+y²))
    新增：支持柏林噪声模式及其参数。
    """
    # 地形公式系数（对应开题报告中的 a, b, c, d, e, f, g）
    a: float = 0.0
    b: float = 1.0
    c: float = 0.5
    d: float = 0.2
    e: float = 0.3
    f: float = 0.8
    g: float = 0.4

    # 地形生成范围（单位：米）
    x_min: float = -50.0
    x_max: float = 50.0
    y_min: float = -50.0
    y_max: float = 50.0

    # 采样分辨率（网格步长，单位：米）
    resolution: float = 1.0

    # 新增：地形生成模式 ("function" 或 "perlin")
    terrain_mode: str = "function"

    # 柏林噪声参数（仅在 terrain_mode="perlin" 时有效）
    perlin_scale: float = 50.0          # 缩放因子
    perlin_octaves: int = 6              # 倍频程数
    perlin_persistence: float = 0.5      # 持续度
    perlin_lacunarity: float = 2.0       # 空隙度
    perlin_seed: int = 42                # 随机种子
    perlin_height_scale: float = 20.0    # 高度缩放

    def to_dict(self) -> dict:
        """转换为字典，便于序列化"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        """从字典创建实例，兼容旧数据（没有新增字段时使用默认值）"""
        # 处理旧数据（没有 terrain_mode 字段）
        if "terrain_mode" not in data:
            data["terrain_mode"] = "function"
        # 如果缺少柏林噪声参数，使用默认值
        if "perlin_scale" not in data:
            data["perlin_scale"] = 50.0
        if "perlin_octaves" not in data:
            data["perlin_octaves"] = 6
        if "perlin_persistence" not in data:
            data["perlin_persistence"] = 0.5
        if "perlin_lacunarity" not in data:
            data["perlin_lacunarity"] = 2.0
        if "perlin_seed" not in data:
            data["perlin_seed"] = 42
        if "perlin_height_scale" not in data:
            data["perlin_height_scale"] = 20.0
        return cls(**data)


@dataclass
class SphereObstacle:
    """球体障碍物模型"""
    # 障碍物类型标识（用于序列化时区分）
    type: str = field(default="sphere", init=False)

    # 球心坐标 (x, y, z) 单位：米
    center_x: float = 0.0
    center_y: float = 0.0
    center_z: float = 0.0

    # 球体半径（单位：米）
    radius: float = 1.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d['type'] = 'sphere'  # 显式添加类型字段
        return d


@dataclass
class BoxObstacle:
    """长方体障碍物模型"""
    # 障碍物类型标识
    type: str = field(default="box", init=False)

    # 几何中心坐标 (x, y, z)
    center_x: float = 0.0
    center_y: float = 0.0
    center_z: float = 0.0

    # 三轴尺寸（长、宽、高），单位：米
    length: float = 1.0   # x轴方向长度
    width: float = 1.0    # y轴方向宽度
    height: float = 1.0   # z轴方向高度

    def to_dict(self) -> dict:
        d = asdict(self)
        d['type'] = 'box'
        return d


# 障碍物类型的联合（用于类型注解）
Obstacle = Union[SphereObstacle, BoxObstacle]


@dataclass
class UAVConfig:
    """无人机配置（质点模型）"""
    # 起点坐标 (x, y, z)
    start_x: float = -40.0
    start_y: float = -40.0
    start_z: float = 15.0

    # 终点坐标 (x, y, z)
    goal_x: float = 40.0
    goal_y: float = 40.0
    goal_z: float = 20.0

    # 物理约束
    max_speed: float = 10.0          # 最大速度 m/s
    max_accel: float = 5.0           # 最大加速度 m/s²
    min_altitude: float = 5.0        # 最低飞行高度 m
    max_altitude: float = 50.0       # 最高飞行高度 m

    # 新增：最小高度和最大高度的独立基准 ("amsl" 或 "agl")
    min_altitude_ref: str = "amsl"
    max_altitude_ref: str = "amsl"
    # 为了兼容旧数据，保留统一的 altitude_ref（新代码中建议不再使用）
    altitude_ref: str = "amsl"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        # 兼容旧数据：如果没有 min_altitude_ref 和 max_altitude_ref，则使用 altitude_ref 填充
        if "min_altitude_ref" not in data:
            unified = data.get("altitude_ref", "amsl")
            data["min_altitude_ref"] = unified
            data["max_altitude_ref"] = unified
        # 确保 altitude_ref 存在（虽然不再使用，但保持 to_dict 完整）
        if "altitude_ref" not in data:
            data["altitude_ref"] = "amsl"
        return cls(**data)


@dataclass
class AlgorithmConfig:
    """AC算法配置参数（包含全局-局部协同机制）"""
    # 超参数
    learning_rate: float = 0.001      # 学习率
    gamma: float = 0.99                # 折扣因子
    batch_size: int = 64               # 批次大小
    hidden_size: int = 64               # 神经网络隐藏层大小
    buffer_capacity: int = 10000        # 经验回放缓冲区容量

    # 奖励函数系数
    reward_dist: float = 1.2           # 距离奖励系数（鼓励接近目标）
    reward_collision: float = -10.0    # 碰撞惩罚
    reward_smooth: float = 0.1         # 平滑奖励（鼓励平稳飞行）

    # 协同机制
    cooperative: bool = True            # 是否启用全局-局部协同机制

    # ===== 全局-局部协同新增参数 =====
    use_global_path: bool = True        # 是否启用A*全局路径引导
    global_step_size: float = 15.0      # 大步长（米）
    cylinder_radius: float = 20.0       # 圆柱体半径（米）
    switch_threshold: float = 5.0       # 切换路径点距离阈值（米）
    reward_path_follow: float = -0.5    # 路径跟随奖励（负值表示偏离惩罚）
    reward_path_progress: float = 1.0   # 路径进度奖励系数
    reward_out_of_cylinder: float = -5.0 # 超出圆柱体惩罚

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        # 兼容旧数据：如果没有 hidden_size 或 buffer_capacity，使用默认值
        if "hidden_size" not in data:
            data["hidden_size"] = 64
        if "buffer_capacity" not in data:
            data["buffer_capacity"] = 10000
        # 兼容旧数据：移除已废弃的 sample_mode 字段
        data.pop("sample_mode", None)
        # 兼容旧数据：协同机制相关字段
        if "use_global_path" not in data:
            data["use_global_path"] = False
        if "global_step_size" not in data:
            data["global_step_size"] = 15.0
        if "cylinder_radius" not in data:
            data["cylinder_radius"] = 20.0
        if "switch_threshold" not in data:
            data["switch_threshold"] = 5.0
        if "reward_path_follow" not in data:
            data["reward_path_follow"] = -0.5
        if "reward_path_progress" not in data:
            data["reward_path_progress"] = 1.0
        if "reward_out_of_cylinder" not in data:
            data["reward_out_of_cylinder"] = -5.0
        return cls(**data)


@dataclass
class TrainConfig:
    """训练配置参数"""
    num_epochs: int = 1000              # 训练轮数
    save_interval: int = 100            # 模型保存间隔（轮）
    log_interval: int = 10              # 日志输出间隔（轮）
    max_steps: int = 500                 # 每回合最大步数（新增）

    # ===== 探索退火参数 =====
    exploration_init: float = 1.5        # 初始探索偏置（加到 log_std 上，训练后期退火到0）
    # ===== 探索退火参数结束 =====

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        # 兼容旧数据：如果没有 max_steps 字段，使用默认值 500
        if "max_steps" not in data:
            data["max_steps"] = 500
        # 兼容旧数据：探索退火参数
        if "exploration_init" not in data:
            data["exploration_init"] = 1.5
        return cls(**data)


@dataclass
class MountainData:
    """
    山区完整数据（地形 + 障碍物列表）
    用于保存/加载山区建模结果
    """
    terrain: TerrainData = field(default_factory=TerrainData)
    obstacles: List[Obstacle] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转换为字典，处理障碍物类型"""
        return {
            'terrain': self.terrain.to_dict(),
            'obstacles': [obs.to_dict() for obs in self.obstacles]
        }

    @classmethod
    def from_dict(cls, data: dict):
        """从字典恢复 MountainData"""
        terrain = TerrainData.from_dict(data['terrain'])
        obstacles = []
        for obs_data in data.get('obstacles', []):
            obs_type = obs_data.get('type')
            if obs_type == 'sphere':
                # 移除 type 字段，因为 SphereObstacle 的 __init__ 不需要它
                clean = {k: v for k, v in obs_data.items() if k != 'type'}
                obstacles.append(SphereObstacle(**clean))
            elif obs_type == 'box':
                clean = {k: v for k, v in obs_data.items() if k != 'type'}
                obstacles.append(BoxObstacle(**clean))
            else:
                raise ValueError(f"Unknown obstacle type: {obs_type}")
        return cls(terrain=terrain, obstacles=obstacles)


# 为了方便，也可以定义组合配置（无人机+算法+训练）用于训练任务
@dataclass
class TrainJobConfig:
    """训练任务完整配置（山区 + 无人机 + 算法 + 训练参数）"""
    mountain: MountainData = field(default_factory=MountainData)
    uav: UAVConfig = field(default_factory=UAVConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    training: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return {
            'mountain': self.mountain.to_dict(),
            'uav': self.uav.to_dict(),
            'algorithm': self.algorithm.to_dict(),
            'training': self.training.to_dict()
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            mountain=MountainData.from_dict(data['mountain']),
            uav=UAVConfig.from_dict(data['uav']),
            algorithm=AlgorithmConfig.from_dict(data['algorithm']),
            training=TrainConfig.from_dict(data['training'])
        )