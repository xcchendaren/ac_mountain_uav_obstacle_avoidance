"""
基于 PyVista 的 3D 渲染器，支持正确的深度遮挡，并添加地形侧面和底面。
侧面沿着地形边缘起伏，动态贴合地形边缘曲线。
可嵌入 PyQt5 主窗口。
新增：点拾取功能，用于方便地获取场景中的位置。
新增：绘制起点和终点（带独立大小控制），无人机大小直接使用半径。
新增：绘制全局路径和斜圆柱体段（用于全局-局部协同机制可视化）。
"""

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from data.models import SphereObstacle, BoxObstacle
from business.uav_builder import UAV


def create_cylinder_between_points(p1, p2, radius, color='cyan', opacity=0.4):
    """
    创建以p1和p2为端点的圆柱体Mesh（用于斜圆柱体）
    p1, p2: 三维点坐标 (x,y,z)
    radius: 圆柱半径
    color: 颜色
    opacity: 透明度
    """
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    vec = p2 - p1
    length = np.linalg.norm(vec)
    if length == 0:
        return None
    direction = vec / length
    cylinder = pv.Cylinder(center=(0, 0, 0), direction=(0, 1, 0),
                           radius=radius, height=length)
    # 旋转
    y_axis = np.array([0, 1, 0])
    rot_axis = np.cross(y_axis, direction)
    dot = np.dot(y_axis, direction)
    if np.isclose(dot, -1.0):
        rot_axis = np.array([1, 0, 0])
        angle = np.pi
    else:
        angle = np.arccos(dot)
    if not np.isclose(angle, 0):
        cylinder = cylinder.rotate_vector(rot_axis, np.degrees(angle), point=(0, 0, 0))
    center = (p1 + p2) / 2
    cylinder = cylinder.translate(center)
    return cylinder


class PyVistaRenderer(QWidget):
    """
    使用 PyVista 的 3D 渲染部件，直接嵌入 PyQt 布局。
    地形绘制时自动添加侧面和底面，形成封闭立体效果。
    支持点拾取模式，用于交互式坐标选择。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # 创建 PyVista Qt 交互器
        self.plotter = QtInteractor(self)
        # 设置布局
        layout = QVBoxLayout()
        layout.addWidget(self.plotter.interactor)
        self.setLayout(layout)

        # 显示坐标轴（方向指示）
        self.plotter.show_axes()
        # 显示带刻度的坐标网格，并保存演员以便后续更新
        self.grid_actor = self.plotter.show_grid()

        # 存储已绘制的对象以便清除
        self.terrain_actor = None
        self.terrain_side_actors = []  # 侧面和底面
        self.obstacle_actors = []
        self.uav_actor = None
        self.trajectory_actor = None
        self.start_actor = None      # 起点标记
        self.goal_actor = None       # 终点标记

        # 全局路径相关
        self.global_path_actor = None
        self.cylinder_actors = []    # 存储圆柱体actor，便于清除

        # 拾取模式相关
        self.pick_mode_active = False
        self.pick_callback = None

    def clear_all(self):
        """清除所有内容"""
        self.plotter.clear()
        self.terrain_actor = None
        self.terrain_side_actors.clear()
        self.obstacle_actors.clear()
        self.uav_actor = None
        self.trajectory_actor = None
        self.start_actor = None
        self.goal_actor = None
        self.global_path_actor = None
        self.cylinder_actors.clear()
        # 清除后重新创建网格，并保存新演员
        self.grid_actor = self.plotter.show_grid()
        # 重置相机
        self.plotter.reset_camera()

    def draw_terrain(self, X, Y, Z, cmap='terrain', opacity=0.8,
                     bottom_offset=5.0, wall_color='gray', reset_camera=True):
        """
        绘制地形曲面，并添加底面和四个垂直面，侧面沿着地形边缘起伏。
        绘制完成后更新坐标网格并可选重置相机。

        Args:
            X, Y, Z: 地形网格坐标 (2D arrays)
            cmap: 地形曲面颜色映射
            opacity: 地形透明度
            bottom_offset: 底部平面相对于地形最低点的偏移量（正值向下）
            wall_color: 侧面和底面的颜色
            reset_camera: 是否重置相机以适应新地形范围
        """
        self.clear_terrain()

        # 1. 绘制地形曲面
        grid = pv.StructuredGrid(X, Y, Z)
        self.terrain_actor = self.plotter.add_mesh(grid, cmap=cmap, opacity=opacity, lighting=True)

        # 2. 计算边界和底部高度
        x = X[0, :]
        y = Y[:, 0]
        nx = len(x)
        ny = len(y)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        z_min = Z.min()
        bottom_z = z_min - bottom_offset

        # 3. 创建底部平面（矩形）
        x_rect = np.array([x_min, x_max])
        y_rect = np.array([y_min, y_max])
        Xb, Yb = np.meshgrid(x_rect, y_rect)
        Zb = np.full_like(Xb, bottom_z)
        bottom_grid = pv.StructuredGrid(Xb, Yb, Zb)
        actor_bottom = self.plotter.add_mesh(bottom_grid, color=wall_color, opacity=opacity*0.5)
        self.terrain_side_actors.append(actor_bottom)

        # 4. 创建四个侧面，使其贴合地形边缘曲线
        # 左面 (x = x_min) - 沿 Y 方向
        n_left = ny
        x_left = np.full((2, n_left), x_min)
        y_left = np.tile(Y[:, 0].reshape(1, -1), (2, 1))
        z_left = np.vstack([np.full(n_left, bottom_z), Z[:, 0]])
        grid_left = pv.StructuredGrid(x_left, y_left, z_left)
        actor_left = self.plotter.add_mesh(grid_left, color=wall_color, opacity=opacity*0.3)
        self.terrain_side_actors.append(actor_left)

        # 右面 (x = x_max)
        n_right = ny
        x_right = np.full((2, n_right), x_max)
        y_right = np.tile(Y[:, -1].reshape(1, -1), (2, 1))
        z_right = np.vstack([np.full(n_right, bottom_z), Z[:, -1]])
        grid_right = pv.StructuredGrid(x_right, y_right, z_right)
        actor_right = self.plotter.add_mesh(grid_right, color=wall_color, opacity=opacity*0.3)
        self.terrain_side_actors.append(actor_right)

        # 前面 (y = y_min)
        n_front = nx
        x_front = np.tile(X[0, :].reshape(1, -1), (2, 1))
        y_front = np.full((2, n_front), y_min)
        z_front = np.vstack([np.full(n_front, bottom_z), Z[0, :]])
        grid_front = pv.StructuredGrid(x_front, y_front, z_front)
        actor_front = self.plotter.add_mesh(grid_front, color=wall_color, opacity=opacity*0.3)
        self.terrain_side_actors.append(actor_front)

        # 后面 (y = y_max)
        n_back = nx
        x_back = np.tile(X[-1, :].reshape(1, -1), (2, 1))
        y_back = np.full((2, n_back), y_max)
        z_back = np.vstack([np.full(n_back, bottom_z), Z[-1, :]])
        grid_back = pv.StructuredGrid(x_back, y_back, z_back)
        actor_back = self.plotter.add_mesh(grid_back, color=wall_color, opacity=opacity*0.3)
        self.terrain_side_actors.append(actor_back)

        # 5. 更新坐标网格（移除旧的，重新创建以匹配新范围）
        if hasattr(self, 'grid_actor') and self.grid_actor is not None:
            self.plotter.remove_actor(self.grid_actor)
        self.grid_actor = self.plotter.show_grid()

        # 6. 根据参数决定是否重置相机
        if reset_camera:
            self.plotter.reset_camera()
        self.plotter.render()

    def clear_terrain(self):
        """清除地形及其侧面/底面"""
        if self.terrain_actor:
            self.plotter.remove_actor(self.terrain_actor)
            self.terrain_actor = None
        for actor in self.terrain_side_actors:
            self.plotter.remove_actor(actor)
        self.terrain_side_actors.clear()

    def draw_obstacle(self, obstacle, color='red', opacity=0.5):
        """绘制单个障碍物"""
        if isinstance(obstacle, SphereObstacle):
            sphere = pv.Sphere(radius=obstacle.radius,
                               center=(obstacle.center_x, obstacle.center_y, obstacle.center_z))
            actor = self.plotter.add_mesh(sphere, color=color, opacity=opacity, smooth_shading=True)
        elif isinstance(obstacle, BoxObstacle):
            bounds = [
                obstacle.center_x - obstacle.length/2, obstacle.center_x + obstacle.length/2,
                obstacle.center_y - obstacle.width/2,  obstacle.center_y + obstacle.width/2,
                obstacle.center_z - obstacle.height/2, obstacle.center_z + obstacle.height/2
            ]
            box = pv.Box(bounds=bounds)
            actor = self.plotter.add_mesh(box, color=color, opacity=opacity, smooth_shading=True)
        else:
            return
        self.obstacle_actors.append(actor)

    def draw_obstacles(self, obstacles, color='red', opacity=0.5):
        """批量绘制障碍物"""
        for obs in obstacles:
            self.draw_obstacle(obs, color, opacity)

    def clear_obstacles(self):
        for actor in self.obstacle_actors:
            self.plotter.remove_actor(actor)
        self.obstacle_actors.clear()

    def draw_uav(self, uav: UAV, color='blue', size=2.0):
        """
        绘制无人机当前位置（球体表示），size 为球体半径（米），不重置相机
        """
        self.clear_uav()
        sphere = pv.Sphere(radius=size, center=tuple(uav.position))
        self.uav_actor = self.plotter.add_mesh(sphere, color=color, smooth_shading=True, reset_camera=False)

    def clear_uav(self):
        if self.uav_actor:
            self.plotter.remove_actor(self.uav_actor)
            self.uav_actor = None

    def draw_start(self, position, size=2.0, color='green'):
        """绘制起点标记（球体），size 为半径（米）"""
        self.clear_start()
        sphere = pv.Sphere(radius=size, center=tuple(position))
        self.start_actor = self.plotter.add_mesh(sphere, color=color, smooth_shading=True, reset_camera=False)

    def clear_start(self):
        if self.start_actor:
            self.plotter.remove_actor(self.start_actor)
            self.start_actor = None

    def draw_goal(self, position, size=2.0, color='red'):
        """绘制终点标记（球体），size 为半径（米）"""
        self.clear_goal()
        sphere = pv.Sphere(radius=size, center=tuple(position))
        self.goal_actor = self.plotter.add_mesh(sphere, color=color, smooth_shading=True, reset_camera=False)

    def clear_goal(self):
        if self.goal_actor:
            self.plotter.remove_actor(self.goal_actor)
            self.goal_actor = None

    def draw_trajectory(self, trajectory, color='green', line_width=3):
        """绘制轨迹线"""
        self.clear_trajectory()
        if len(trajectory) < 2:
            return
        line = pv.lines_from_points(trajectory)
        self.trajectory_actor = self.plotter.add_mesh(line, color=color, line_width=line_width)

    def clear_trajectory(self):
        if self.trajectory_actor:
            self.plotter.remove_actor(self.trajectory_actor)
            self.trajectory_actor = None

    def set_background(self, color='white'):
        self.plotter.set_background(color)

    # ---------- 全局路径与圆柱体可视化 ----------
    def draw_global_path(self, waypoints, color='red', line_width=3):
        """
        绘制全局路径（折线）
        waypoints: (N, 3) numpy数组，路径点坐标
        """
        self.clear_global_path()
        if len(waypoints) < 2:
            return
        lines = pv.PolyData(waypoints)
        lines.lines = np.hstack([[len(waypoints)], np.arange(len(waypoints))]).astype(np.int64)
        self.global_path_actor = self.plotter.add_mesh(lines, color=color, line_width=line_width,
                                                        render_lines_as_tubes=True, reset_camera=False)

    def clear_global_path(self):
        if self.global_path_actor:
            self.plotter.remove_actor(self.global_path_actor)
            self.global_path_actor = None

    def draw_cylinder_segments(self, waypoints, radius, color='cyan', opacity=0.3):
        """
        绘制沿路径的斜圆柱体段（每两个相邻路径点之间一个圆柱体）
        waypoints: (N, 3) 路径点坐标
        radius: 圆柱半径
        color: 颜色
        opacity: 透明度
        """
        self.clear_cylinders()
        if len(waypoints) < 2:
            return
        for i in range(len(waypoints)-1):
            cyl = create_cylinder_between_points(waypoints[i], waypoints[i+1],
                                                 radius=radius, color=color, opacity=opacity)
            if cyl:
                actor = self.plotter.add_mesh(cyl, color=color, opacity=opacity, show_edges=False,
                                              reset_camera=False)
                self.cylinder_actors.append(actor)

    def clear_cylinders(self):
        for actor in self.cylinder_actors:
            self.plotter.remove_actor(actor)
        self.cylinder_actors.clear()

    # ---------- 拾取模式相关 ----------
    def enable_pick_mode(self, callback):
        """
        启用点拾取模式，并设置回调函数。
        回调函数将接收一个长度为3的numpy数组作为拾取点的坐标。
        """
        if self.pick_mode_active:
            self.disable_pick_mode()
        self.pick_callback = callback
        self.pick_mode_active = True
        self.plotter.enable_point_picking(callback=self._on_point_picked, show_message=False,
                                          picker='point', left_clicking=True)

    def disable_pick_mode(self):
        """禁用点拾取模式"""
        if self.pick_mode_active:
            self.plotter.disable_picking()
            self.pick_mode_active = False
            self.pick_callback = None

    def _on_point_picked(self, point):
        """内部拾取回调，提取坐标并调用用户回调"""
        if self.pick_callback is not None:
            # point 可能是长度为3的列表或元组
            self.pick_callback(np.array(point, dtype=float))
        # 拾取一次后自动禁用，避免干扰（可根据需要调整）
        self.disable_pick_mode()

    def show(self):
        self.plotter.show()