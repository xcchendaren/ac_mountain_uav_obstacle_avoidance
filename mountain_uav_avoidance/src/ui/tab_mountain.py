"""
山区设置标签页：包含地形参数与障碍物配置，支持保存/加载
新增：柏林噪声地形生成，支持完整的参数调节；增加“随机参数”按钮；增加障碍物位置拾取功能。
修改：生成地形、添加/删除障碍物时自动清除轨迹。
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QLineEdit, QComboBox, QListWidget,
    QFileDialog, QMessageBox, QInputDialog
)
from PyQt5.QtCore import Qt
import os
import random
import numpy as np

from business.terrain_builder import generate_terrain, generate_terrain_perlin
from business.obstacle_builder import create_sphere_obstacle, create_box_obstacle
from data.models import TerrainData, MountainData, SphereObstacle, BoxObstacle
from data.serializer import save_json, load_json


class TabMountain(QWidget):
    """山区设置标签页：包含地形参数与障碍物配置，支持保存/加载"""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.current_mountain_data = MountainData()
        self.current_mountain_name = None  # 当前加载的山区名称
        self.last_terrain_X = None         # 缓存最近生成的地形网格
        self.last_terrain_Y = None
        self.last_terrain_Z = None
        self.init_ui()
        self._update_obstacle_inputs()  # 初始化障碍物输入框可见性
        self._on_terrain_mode_changed(0)  # 默认显示传统函数参数

    def init_ui(self):
        layout = QVBoxLayout(self)

        # ========== 山区地形设置组 ==========
        group_terrain = QGroupBox("山区地形设置")
        terrain_layout = QVBoxLayout(group_terrain)

        # --- 地形生成模式切换 ---
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("生成模式:"))
        self.cb_terrain_mode = QComboBox()
        self.cb_terrain_mode.addItems(["传统函数", "柏林噪声"])
        self.cb_terrain_mode.currentIndexChanged.connect(self._on_terrain_mode_changed)
        mode_layout.addWidget(self.cb_terrain_mode)
        terrain_layout.addLayout(mode_layout)

        # ===== 传统函数参数组 =====
        self.func_group = QWidget()
        func_layout = QVBoxLayout(self.func_group)
        func_layout.setContentsMargins(0, 0, 0, 0)

        # 第一行系数 a,b,c
        form_layout = QHBoxLayout()
        form_layout.addWidget(QLabel("a:"))
        self.edit_a = QLineEdit("0.5")
        form_layout.addWidget(self.edit_a)
        form_layout.addWidget(QLabel("b:"))
        self.edit_b = QLineEdit("1.2")
        form_layout.addWidget(self.edit_b)
        form_layout.addWidget(QLabel("c:"))
        self.edit_c = QLineEdit("0.8")
        form_layout.addWidget(self.edit_c)
        func_layout.addLayout(form_layout)

        # 第二行系数 d,e,f,g
        form_layout2 = QHBoxLayout()
        form_layout2.addWidget(QLabel("d:"))
        self.edit_d = QLineEdit("0.3")
        form_layout2.addWidget(self.edit_d)
        form_layout2.addWidget(QLabel("e:"))
        self.edit_e = QLineEdit("0.6")
        form_layout2.addWidget(self.edit_e)
        form_layout2.addWidget(QLabel("f:"))
        self.edit_f = QLineEdit("1.5")
        form_layout2.addWidget(self.edit_f)
        form_layout2.addWidget(QLabel("g:"))
        self.edit_g = QLineEdit("0.4")
        form_layout2.addWidget(self.edit_g)
        func_layout.addLayout(form_layout2)

        terrain_layout.addWidget(self.func_group)

        # ===== 柏林噪声参数组 =====
        self.noise_group = QWidget()
        noise_layout = QVBoxLayout(self.noise_group)
        noise_layout.setContentsMargins(0, 0, 0, 0)

        # 第一行参数
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("scale (缩放):"))
        self.edit_scale = QLineEdit("50.0")
        row1.addWidget(self.edit_scale)
        row1.addWidget(QLabel("octaves (倍频程):"))
        self.edit_octaves = QLineEdit("6")
        row1.addWidget(self.edit_octaves)
        noise_layout.addLayout(row1)

        # 第二行参数
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("persistence (持续度):"))
        self.edit_persistence = QLineEdit("0.5")
        row2.addWidget(self.edit_persistence)
        row2.addWidget(QLabel("lacunarity (空隙度):"))
        self.edit_lacunarity = QLineEdit("2.0")
        row2.addWidget(self.edit_lacunarity)
        noise_layout.addLayout(row2)

        # 第三行参数
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("seed (随机种子):"))
        self.edit_seed = QLineEdit("42")
        row3.addWidget(self.edit_seed)
        row3.addWidget(QLabel("height_scale (高度缩放):"))
        self.edit_height_scale = QLineEdit("20.0")
        row3.addWidget(self.edit_height_scale)
        noise_layout.addLayout(row3)

        terrain_layout.addWidget(self.noise_group)

        # --- 公共范围设置（两种模式共用）---
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("X范围:"))
        self.edit_xmin = QLineEdit("-50")
        self.edit_xmax = QLineEdit("50")
        size_layout.addWidget(self.edit_xmin)
        size_layout.addWidget(QLabel("~"))
        size_layout.addWidget(self.edit_xmax)
        size_layout.addWidget(QLabel("Y范围:"))
        self.edit_ymin = QLineEdit("-50")
        self.edit_ymax = QLineEdit("50")
        size_layout.addWidget(self.edit_ymin)
        size_layout.addWidget(QLabel("~"))
        size_layout.addWidget(self.edit_ymax)
        terrain_layout.addLayout(size_layout)

        # --- 随机参数按钮（放在生成按钮上方）---
        random_layout = QHBoxLayout()
        random_layout.addStretch()
        self.btn_random = QPushButton("随机参数")
        self.btn_random.clicked.connect(self._on_random_parameters)
        random_layout.addWidget(self.btn_random)
        terrain_layout.addLayout(random_layout)

        # 生成地形按钮
        btn_gen_terrain = QPushButton("生成地形")
        btn_gen_terrain.clicked.connect(self._on_gen_terrain)
        terrain_layout.addWidget(btn_gen_terrain)

        layout.addWidget(group_terrain)

        # ========== 障碍物设置组 ==========
        group_obstacle = QGroupBox("障碍物设置")
        obstacle_layout = QVBoxLayout(group_obstacle)

        # 类型选择
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("类型:"))
        self.cb_obstacle_type = QComboBox()
        self.cb_obstacle_type.addItems(["球体", "长方体"])
        self.cb_obstacle_type.currentIndexChanged.connect(self._update_obstacle_inputs)
        type_layout.addWidget(self.cb_obstacle_type)
        obstacle_layout.addLayout(type_layout)

        # 位置输入（带拾取按钮）
        pos_layout = QHBoxLayout()
        pos_layout.addWidget(QLabel("X:"))
        self.edit_obs_x = QLineEdit("0")
        pos_layout.addWidget(self.edit_obs_x)
        pos_layout.addWidget(QLabel("Y:"))
        self.edit_obs_y = QLineEdit("0")
        pos_layout.addWidget(self.edit_obs_y)
        pos_layout.addWidget(QLabel("Z:"))
        self.edit_obs_z = QLineEdit("5")
        pos_layout.addWidget(self.edit_obs_z)
        # 添加拾取位置按钮
        self.btn_pick_pos = QPushButton("拾取位置")
        self.btn_pick_pos.clicked.connect(self._on_pick_position)
        pos_layout.addWidget(self.btn_pick_pos)
        obstacle_layout.addLayout(pos_layout)

        # 尺寸输入
        size_obs_layout = QHBoxLayout()
        self.label_obs_r = QLabel("半径:")
        size_obs_layout.addWidget(self.label_obs_r)
        self.edit_obs_r = QLineEdit("3")
        size_obs_layout.addWidget(self.edit_obs_r)
        self.label_obs_w = QLabel("宽:")
        size_obs_layout.addWidget(self.label_obs_w)
        self.edit_obs_w = QLineEdit("4")
        size_obs_layout.addWidget(self.edit_obs_w)
        self.label_obs_h = QLabel("高:")
        size_obs_layout.addWidget(self.label_obs_h)
        self.edit_obs_h = QLineEdit("5")
        size_obs_layout.addWidget(self.edit_obs_h)
        obstacle_layout.addLayout(size_obs_layout)

        btn_add_obs = QPushButton("添加障碍")
        btn_add_obs.clicked.connect(self._on_add_obstacle)
        obstacle_layout.addWidget(btn_add_obs)

        self.obstacle_list = QListWidget()
        self.obstacle_list.itemDoubleClicked.connect(self._on_remove_obstacle)
        obstacle_layout.addWidget(self.obstacle_list)

        layout.addWidget(group_obstacle)

        # ========== 操作按钮 ==========
        btn_layout = QHBoxLayout()
        btn_save = QPushButton("保存山区")
        btn_load = QPushButton("加载山区")
        btn_delete = QPushButton("删除山区")
        btn_save.clicked.connect(self._on_save_mountain)
        btn_load.clicked.connect(self._on_load_mountain)
        btn_delete.clicked.connect(self._on_delete_mountain)
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_load)
        btn_layout.addWidget(btn_delete)
        layout.addLayout(btn_layout)

        layout.addStretch()

    # ---------- 辅助方法 ----------
    def _update_obstacle_inputs(self):
        """根据当前选择的障碍物类型，显示对应的尺寸输入框"""
        is_sphere = self.cb_obstacle_type.currentText() == "球体"
        self.label_obs_r.setText("半径:" if is_sphere else "长:")
        self.label_obs_w.setVisible(not is_sphere)
        self.edit_obs_w.setVisible(not is_sphere)
        self.label_obs_h.setVisible(not is_sphere)
        self.edit_obs_h.setVisible(not is_sphere)

    def _on_terrain_mode_changed(self, index):
        """切换地形生成模式时，显示/隐藏对应的参数组"""
        if index == 0:  # 传统函数
            self.func_group.show()
            self.noise_group.hide()
        else:           # 柏林噪声
            self.func_group.hide()
            self.noise_group.show()

    # ---------- 随机参数（不生成地形）----------
    def _on_random_parameters(self):
        """随机生成当前模式下的参数，仅更新输入框，不生成地形"""
        mode = self.cb_terrain_mode.currentIndex()
        if mode == 0:
            # 随机传统函数参数
            self.edit_a.setText(f"{random.uniform(-2, 2):.3f}")
            self.edit_b.setText(f"{random.uniform(0.5, 2.5):.3f}")
            self.edit_c.setText(f"{random.uniform(0, 3):.3f}")
            self.edit_d.setText(f"{random.uniform(0.1, 0.6):.3f}")
            self.edit_e.setText(f"{random.uniform(0, 2):.3f}")
            self.edit_f.setText(f"{random.uniform(0.5, 3):.3f}")
            self.edit_g.setText(f"{random.uniform(0.1, 0.6):.3f}")
        else:
            # 随机柏林噪声参数
            self.edit_scale.setText(f"{random.uniform(20, 100):.1f}")
            self.edit_octaves.setText(str(random.randint(3, 8)))
            self.edit_persistence.setText(f"{random.uniform(0.3, 0.8):.3f}")
            self.edit_lacunarity.setText(f"{random.uniform(2.0, 3.0):.3f}")
            self.edit_seed.setText(str(random.randint(0, 9999)))
            self.edit_height_scale.setText(f"{random.uniform(10, 30):.1f}")

        self.main_window.log("参数已随机生成，点击“生成地形”查看效果")

    # ---------- 障碍物位置拾取 ----------
    def _on_pick_position(self):
        """进入拾取模式，点击场景自动填充坐标"""
        if not hasattr(self.main_window, 'renderer'):
            QMessageBox.warning(self, "警告", "渲染器未初始化")
            return
        self.main_window.log("请在3D视图中点击选择障碍物位置")
        self.btn_pick_pos.setEnabled(False)
        self.btn_pick_pos.setText("拾取中...")
        self.main_window.renderer.enable_pick_mode(self._on_position_picked)

    def _on_position_picked(self, point):
        """拾取回调，将坐标填入输入框"""
        self.edit_obs_x.setText(f"{point[0]:.2f}")
        self.edit_obs_y.setText(f"{point[1]:.2f}")
        self.edit_obs_z.setText(f"{point[2]:.2f}")
        self.btn_pick_pos.setEnabled(True)
        self.btn_pick_pos.setText("拾取位置")
        self.main_window.log(f"已拾取位置: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})")

    # ---------- 地形生成 ----------
    def _on_gen_terrain(self):
        try:
            # 获取共同参数
            x_min = float(self.edit_xmin.text())
            x_max = float(self.edit_xmax.text())
            y_min = float(self.edit_ymin.text())
            y_max = float(self.edit_ymax.text())

            terrain_data = TerrainData(
                x_min=x_min, x_max=x_max,
                y_min=y_min, y_max=y_max,
                resolution=1.0
            )

            if self.cb_terrain_mode.currentIndex() == 0:
                # 传统函数模式
                a = float(self.edit_a.text())
                b = float(self.edit_b.text())
                c = float(self.edit_c.text())
                d = float(self.edit_d.text())
                e = float(self.edit_e.text())
                f = float(self.edit_f.text())
                g = float(self.edit_g.text())
                terrain_data.a = a
                terrain_data.b = b
                terrain_data.c = c
                terrain_data.d = d
                terrain_data.e = e
                terrain_data.f = f
                terrain_data.g = g
                terrain_data.terrain_mode = "function"

                X, Y, Z = generate_terrain(terrain_data)
                self.current_mountain_data.terrain = terrain_data
                self.last_terrain_X, self.last_terrain_Y, self.last_terrain_Z = X, Y, Z
                # 清除轨迹
                self.main_window.renderer.clear_trajectory()
                self.main_window.render_terrain(X, Y, Z)
                self.main_window.log(f"传统函数地形生成成功，高度范围 {Z.min():.2f} ~ {Z.max():.2f}")
            else:
                # 柏林噪声模式
                scale = float(self.edit_scale.text())
                octaves = int(self.edit_octaves.text())
                persistence = float(self.edit_persistence.text())
                lacunarity = float(self.edit_lacunarity.text())
                seed = int(self.edit_seed.text()) if self.edit_seed.text().strip() else None
                height_scale = float(self.edit_height_scale.text())

                # 保存参数到 terrain_data
                terrain_data.terrain_mode = "perlin"
                terrain_data.perlin_scale = scale
                terrain_data.perlin_octaves = octaves
                terrain_data.perlin_persistence = persistence
                terrain_data.perlin_lacunarity = lacunarity
                terrain_data.perlin_seed = seed if seed is not None else 0
                terrain_data.perlin_height_scale = height_scale

                X, Y, Z = generate_terrain_perlin(
                    terrain_data,
                    scale=scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    seed=seed,
                    height_scale=height_scale
                )
                self.current_mountain_data.terrain = terrain_data
                self.last_terrain_X, self.last_terrain_Y, self.last_terrain_Z = X, Y, Z
                # 清除轨迹
                self.main_window.renderer.clear_trajectory()
                self.main_window.render_terrain(X, Y, Z)
                self.main_window.log(f"柏林噪声地形生成成功，高度范围 {Z.min():.2f} ~ {Z.max():.2f}")

        except Exception as e:
            self.main_window.log(f"地形生成出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"地形生成失败：{str(e)}")

    # ---------- 障碍物添加/删除 ----------
    def _on_add_obstacle(self):
        try:
            x = float(self.edit_obs_x.text())
            y = float(self.edit_obs_y.text())
            z = float(self.edit_obs_z.text())
            obs_type = self.cb_obstacle_type.currentText()

            if obs_type == "球体":
                radius = float(self.edit_obs_r.text())
                obstacle = create_sphere_obstacle(x, y, z, radius)
                desc = f"球体 @ ({x},{y},{z}) r={radius}"
            else:  # 长方体
                length = float(self.edit_obs_r.text())
                width = float(self.edit_obs_w.text())
                height = float(self.edit_obs_h.text())
                obstacle = create_box_obstacle(x, y, z, length, width, height)
                desc = f"长方体 @ ({x},{y},{z}) 尺寸={length}x{width}x{height}"

            self.current_mountain_data.obstacles.append(obstacle)
            self.obstacle_list.addItem(desc)
            # 清除轨迹
            self.main_window.renderer.clear_trajectory()
            self.main_window.render_obstacle(obstacle)
            self.main_window.log(f"添加障碍: {desc}")
        except Exception as e:
            self.main_window.log(f"添加障碍出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"添加障碍失败：{str(e)}")

    def _on_remove_obstacle(self, item):
        """双击删除障碍物，弹出确认对话框"""
        row = self.obstacle_list.row(item)
        if 0 <= row < len(self.current_mountain_data.obstacles):
            reply = QMessageBox.question(
                self,
                "确认删除",
                f"确定要删除选中的障碍物吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                # 从数据中移除
                del self.current_mountain_data.obstacles[row]
                self.obstacle_list.takeItem(row)
                self.main_window.log("障碍物已移除")

                # 清除轨迹
                self.main_window.renderer.clear_trajectory()
                # 仅重新绘制障碍物（保留地形和相机视角）
                self.main_window.renderer.clear_obstacles()
                for obs in self.current_mountain_data.obstacles:
                    self.main_window.render_obstacle(obs)

    # ---------- 保存/加载 ----------
    def _on_save_mountain(self):
        """保存山区数据，使用 FileManager 标准化路径"""
        mountain_name, ok = QInputDialog.getText(self, "保存山区", "请输入山区名称（如：山区1）:")
        if not ok or not mountain_name.strip():
            return
        mountain_name = mountain_name.strip()
        file_path = self.main_window.file_manager.get_mountain_data_path(mountain_name)
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            save_json(self.current_mountain_data.to_dict(), file_path)
            self.current_mountain_name = mountain_name
            self.main_window.log(f"山区数据已保存至: {file_path}")
            QMessageBox.information(self, "成功", f"山区数据已保存到：{file_path}")
        except Exception as e:
            self.main_window.log(f"保存失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"保存山区数据失败：{str(e)}")

    def _on_load_mountain(self):
        """加载山区：选择山区文件夹，自动读取其下的山区建模数据.json"""
        root = self.main_window.file_manager.get_root_dir()
        dir_path = QFileDialog.getExistingDirectory(self, "选择山区文件夹", root)
        if not dir_path:
            return
        mountain_name = os.path.basename(dir_path)
        file_path = os.path.join(dir_path, "山区建模数据.json")
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "警告", f"所选文件夹中未找到山区建模数据.json")
            return
        try:
            data = load_json(file_path)
            self.current_mountain_data = MountainData.from_dict(data)
            self.current_mountain_name = mountain_name

            # 更新UI控件
            t = self.current_mountain_data.terrain
            self.edit_a.setText(str(t.a))
            self.edit_b.setText(str(t.b))
            self.edit_c.setText(str(t.c))
            self.edit_d.setText(str(t.d))
            self.edit_e.setText(str(t.e))
            self.edit_f.setText(str(t.f))
            self.edit_g.setText(str(t.g))
            self.edit_xmin.setText(str(t.x_min))
            self.edit_xmax.setText(str(t.x_max))
            self.edit_ymin.setText(str(t.y_min))
            self.edit_ymax.setText(str(t.y_max))

            # 根据模式切换下拉框
            if t.terrain_mode == "perlin":
                self.cb_terrain_mode.setCurrentIndex(1)
                # 填充柏林噪声参数
                self.edit_scale.setText(str(t.perlin_scale))
                self.edit_octaves.setText(str(t.perlin_octaves))
                self.edit_persistence.setText(str(t.perlin_persistence))
                self.edit_lacunarity.setText(str(t.perlin_lacunarity))
                self.edit_seed.setText(str(t.perlin_seed))
                self.edit_height_scale.setText(str(t.perlin_height_scale))
            else:
                self.cb_terrain_mode.setCurrentIndex(0)

            # 更新障碍物列表
            self.obstacle_list.clear()
            for obs in self.current_mountain_data.obstacles:
                if isinstance(obs, SphereObstacle):
                    desc = f"球体 @ ({obs.center_x},{obs.center_y},{obs.center_z}) r={obs.radius}"
                else:
                    desc = f"长方体 @ ({obs.center_x},{obs.center_y},{obs.center_z}) 尺寸={obs.length}x{obs.width}x{obs.height}"
                self.obstacle_list.addItem(desc)

            # 根据模式重新生成地形并渲染
            if t.terrain_mode == "perlin":
                X, Y, Z = generate_terrain_perlin(
                    t,
                    scale=t.perlin_scale,
                    octaves=t.perlin_octaves,
                    persistence=t.perlin_persistence,
                    lacunarity=t.perlin_lacunarity,
                    seed=t.perlin_seed,
                    height_scale=t.perlin_height_scale
                )
            else:
                X, Y, Z = generate_terrain(t)
            self.last_terrain_X, self.last_terrain_Y, self.last_terrain_Z = X, Y, Z
            # 清除轨迹
            self.main_window.renderer.clear_trajectory()
            self.main_window.render_terrain(X, Y, Z)
            for obs in self.current_mountain_data.obstacles:
                self.main_window.render_obstacle(obs)

            self.main_window.log(f"已加载山区数据: {file_path}")
            QMessageBox.information(self, "成功", f"山区数据已加载：{mountain_name}")
        except Exception as e:
            self.main_window.log(f"加载失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"加载山区数据失败：{str(e)}")

    def _on_delete_mountain(self):
        """删除当前山区（仅清空内存，不删除磁盘文件）"""
        if self.current_mountain_data is None:
            return
        reply = QMessageBox.question(
            self,
            "确认清空",
            "确定要清空当前山区数据吗？此操作不会删除磁盘文件。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.current_mountain_data = MountainData()
            self.obstacle_list.clear()
            self.main_window.clear_visualization()
            self.last_terrain_X = self.last_terrain_Y = self.last_terrain_Z = None
            self.current_mountain_name = None
            self.main_window.log("山区数据已清空")