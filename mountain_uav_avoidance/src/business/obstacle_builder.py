"""
障碍物建模模块

提供创建球体和长方体障碍物的工厂函数，以及基础碰撞检测函数。
"""

from data.models import SphereObstacle, BoxObstacle
import numpy as np


def create_sphere_obstacle(center_x, center_y, center_z, radius):
    """
    创建球体障碍物
    """
    return SphereObstacle(
        center_x=center_x,
        center_y=center_y,
        center_z=center_z,
        radius=radius
    )


def create_box_obstacle(center_x, center_y, center_z, length, width, height):
    """
    创建长方体障碍物
    """
    return BoxObstacle(
        center_x=center_x,
        center_y=center_y,
        center_z=center_z,
        length=length,
        width=width,
        height=height
    )


def check_collision(uav_pos, obstacle):
    """
    检测无人机是否与单个障碍物发生碰撞

    Args:
        uav_pos: 无人机位置 (x, y, z)
        obstacle: 障碍物对象
    Returns:
        bool: 是否碰撞
    """
    x_u, y_u, z_u = uav_pos
    if isinstance(obstacle, SphereObstacle):
        dx = x_u - obstacle.center_x
        dy = y_u - obstacle.center_y
        dz = z_u - obstacle.center_z
        return (dx*dx + dy*dy + dz*dz) <= obstacle.radius**2
    elif isinstance(obstacle, BoxObstacle):
        return (abs(x_u - obstacle.center_x) <= obstacle.length/2 and
                abs(y_u - obstacle.center_y) <= obstacle.width/2 and
                abs(z_u - obstacle.center_z) <= obstacle.height/2)
    return False


def check_collision_with_list(uav_pos, obstacles):
    """
    检测无人机是否与障碍物列表中的任意一个发生碰撞
    """
    for obs in obstacles:
        if check_collision(uav_pos, obs):
            return True
    return False


if __name__ == "__main__":
    sphere = create_sphere_obstacle(0, 0, 5, 3)
    box = create_box_obstacle(10, 10, 10, 4, 4, 8)
    pos = (1, 1, 6)
    print(f"碰撞检测(球体): {check_collision(pos, sphere)}")
    print(f"碰撞检测(长方体): {check_collision(pos, box)}")
    obs_list = [sphere, box]
    print(f"与列表碰撞: {check_collision_with_list(pos, obs_list)}")