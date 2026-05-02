"""
无人机建模模块

基于质点模型实现 UAV 类，封装位置、速度、加速度的更新，
并提供物理约束校验。
"""

import numpy as np
from data.models import UAVConfig


class UAV:
    """无人机质点模型类"""

    def __init__(self, config: UAVConfig):
        self.config = config
        self.position = np.array([config.start_x, config.start_y, config.start_z], dtype=float)
        self.velocity = np.zeros(3, dtype=float)
        self.acceleration = np.zeros(3, dtype=float)
        self.goal = np.array([config.goal_x, config.goal_y, config.goal_z], dtype=float)
        self.max_speed = config.max_speed
        self.max_accel = config.max_accel
        self.min_altitude = config.min_altitude
        self.max_altitude = config.max_altitude

    def set_acceleration(self, accel):
        self.acceleration = np.array(accel, dtype=float)

    def update(self, dt):
        # 更新速度
        self.velocity += self.acceleration * dt
        # 速度上限限制：如果速度超过最大值，缩放到最大值
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed
        # 更新位置
        self.position += self.velocity * dt

    def check_physical_constraints(self):
        """检查物理约束是否满足（注意：超速已被限制，但保留检查逻辑）"""
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            return False
        accel = np.linalg.norm(self.acceleration)
        if accel > self.max_accel:
            return False
        if self.position[2] < self.min_altitude or self.position[2] > self.max_altitude:
            return False
        return True

    def is_speed_exceeded(self):
        """检查是否超速（速度超过最大速度）—— 此方法将被移除，但暂时保留以防其他依赖"""
        return np.linalg.norm(self.velocity) > self.max_speed

    def get_state(self):
        return np.concatenate([self.position, self.velocity, self.acceleration])

    def reset(self):
        self.position = np.array([self.config.start_x, self.config.start_y, self.config.start_z], dtype=float)
        self.velocity = np.zeros(3, dtype=float)
        self.acceleration = np.zeros(3, dtype=float)

    def distance_to_goal(self):
        return np.linalg.norm(self.position - self.goal)


if __name__ == "__main__":
    from data.models import UAVConfig
    config = UAVConfig()
    uav = UAV(config)
    uav.set_acceleration([1.0, 0.5, 0.2])
    uav.update(0.1)
    print("约束检查:", uav.check_physical_constraints())