"""
训练配置页：包含山区加载、无人机/算法参数配置、协同机制参数、配置文件保存/加载/删除。
对外暴露 get_job_config() → TrainJobConfig，供训练页直接调用。
"""

import os
from pathlib import Path

import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QLineEdit, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QInputDialog, QGridLayout
)
from PyQt5.QtCore import Qt, QSettings

from data.models import (
    MountainData, UAVConfig,
    AlgorithmConfig, TrainConfig, TrainJobConfig
)
from data.serializer import load_json, save_json
from business.terrain_builder import generate_terrain, generate_terrain_perlin


class TabConfig(QWidget):
    """训练配置页：参数填写、配置保存/加载，不含训练运行逻辑"""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.current_mountain_data = MountainData()
        self.current_mountain_name = None
        self.current_config_name = None

        self.init_ui()
        self._connect_signals()

    # =========================================================
    # UI 构建
    # =========================================================
    def init_ui(self):
        layout = QVBoxLayout(self)

        # 1. 加载山区模型组
        group_load = QGroupBox("加载山区模型")
        load_layout = QHBoxLayout(group_load)
        self.btn_load_mountain = QPushButton("选择并加载山区文件夹")
        self.btn_load_mountain.clicked.connect(self._on_load_mountain)
        load_layout.addWidget(self.btn_load_mountain)
        self.label_mountain_info = QLabel("未加载山区数据")
        load_layout.addWidget(self.label_mountain_info)
        layout.addWidget(group_load)

        # 2. 无人机设置组
        group_uav = QGroupBox("无人机设置")
        uav_layout = QVBoxLayout(group_uav)

        # 起点布局
        start_layout = QHBoxLayout()
        start_layout.addWidget(QLabel("起点 X:"))
        self.edit_start_x = QLineEdit("-40")
        start_layout.addWidget(self.edit_start_x)
        start_layout.addWidget(QLabel("Y:"))
        self.edit_start_y = QLineEdit("-40")
        start_layout.addWidget(self.edit_start_y)
        start_layout.addWidget(QLabel("Z:"))
        self.edit_start_z = QLineEdit("15")
        start_layout.addWidget(self.edit_start_z)
        self.btn_pick_start = QPushButton("拾取起点")
        self.btn_pick_start.clicked.connect(self._on_pick_start)
        start_layout.addWidget(self.btn_pick_start)
        uav_layout.addLayout(start_layout)

        # 终点布局
        end_layout = QHBoxLayout()
        end_layout.addWidget(QLabel("终点 X:"))
        self.edit_goal_x = QLineEdit("40")
        end_layout.addWidget(self.edit_goal_x)
        end_layout.addWidget(QLabel("Y:"))
        self.edit_goal_y = QLineEdit("40")
        end_layout.addWidget(self.edit_goal_y)
        end_layout.addWidget(QLabel("Z:"))
        self.edit_goal_z = QLineEdit("20")
        end_layout.addWidget(self.edit_goal_z)
        self.btn_pick_goal = QPushButton("拾取终点")
        self.btn_pick_goal.clicked.connect(self._on_pick_goal)
        end_layout.addWidget(self.btn_pick_goal)
        uav_layout.addLayout(end_layout)

        # 物理约束布局
        constraint_layout = QHBoxLayout()
        constraint_layout.addWidget(QLabel("最大速度:"))
        self.edit_max_speed = QLineEdit("10")
        constraint_layout.addWidget(self.edit_max_speed)
        constraint_layout.addWidget(QLabel("最大加速度:"))
        self.edit_max_accel = QLineEdit("5")
        constraint_layout.addWidget(self.edit_max_accel)

        constraint_layout.addWidget(QLabel("最低高度:"))
        self.edit_min_alt = QLineEdit("5")
        constraint_layout.addWidget(self.edit_min_alt)
        self.cb_min_alt_ref = QComboBox()
        self.cb_min_alt_ref.addItem("海拔", "amsl")
        self.cb_min_alt_ref.addItem("离地", "agl")
        self.cb_min_alt_ref.setCurrentIndex(0)
        constraint_layout.addWidget(self.cb_min_alt_ref)

        constraint_layout.addWidget(QLabel("最高高度:"))
        self.edit_max_alt = QLineEdit("50")
        constraint_layout.addWidget(self.edit_max_alt)
        self.cb_max_alt_ref = QComboBox()
        self.cb_max_alt_ref.addItem("海拔", "amsl")
        self.cb_max_alt_ref.addItem("离地", "agl")
        self.cb_max_alt_ref.setCurrentIndex(0)
        constraint_layout.addWidget(self.cb_max_alt_ref)

        uav_layout.addLayout(constraint_layout)
        layout.addWidget(group_uav)

        # 3. 算法设置组
        group_alg = QGroupBox("算法设置")
        alg_layout = QVBoxLayout(group_alg)

        hyper_layout = QHBoxLayout()
        hyper_layout.addWidget(QLabel("学习率:"))
        self.edit_lr = QLineEdit("0.001")
        hyper_layout.addWidget(self.edit_lr)
        hyper_layout.addWidget(QLabel("折扣因子 γ:"))
        self.edit_gamma = QLineEdit("0.99")
        hyper_layout.addWidget(self.edit_gamma)
        hyper_layout.addWidget(QLabel("批次大小:"))
        self.edit_batch = QSpinBox()
        self.edit_batch.setRange(1, 1024)
        self.edit_batch.setValue(64)
        hyper_layout.addWidget(self.edit_batch)
        alg_layout.addLayout(hyper_layout)

        hidden_layout = QHBoxLayout()
        hidden_layout.addWidget(QLabel("隐藏层大小:"))
        self.edit_hidden_size = QSpinBox()
        self.edit_hidden_size.setRange(16, 512)
        self.edit_hidden_size.setValue(64)
        self.edit_hidden_size.setSingleStep(16)
        hidden_layout.addWidget(self.edit_hidden_size)
        hidden_layout.addWidget(QLabel("经验回放容量:"))
        self.edit_buffer_capacity = QSpinBox()
        self.edit_buffer_capacity.setRange(1000, 100000)
        self.edit_buffer_capacity.setValue(10000)
        self.edit_buffer_capacity.setSingleStep(1000)
        hidden_layout.addWidget(self.edit_buffer_capacity)
        alg_layout.addLayout(hidden_layout)

        reward_layout = QHBoxLayout()
        reward_layout.addWidget(QLabel("距离奖励系数:"))
        self.edit_r_dist = QLineEdit("1.0")
        reward_layout.addWidget(self.edit_r_dist)
        reward_layout.addWidget(QLabel("碰撞惩罚:"))
        self.edit_r_collision = QLineEdit("-10.0")
        reward_layout.addWidget(self.edit_r_collision)
        reward_layout.addWidget(QLabel("平滑奖励:"))
        self.edit_r_smooth = QLineEdit("0.1")
        reward_layout.addWidget(self.edit_r_smooth)
        alg_layout.addLayout(reward_layout)

        self.cb_coop = QCheckBox("启用全局-局部协同机制")
        self.cb_coop.setChecked(True)
        alg_layout.addWidget(self.cb_coop)

        # 协同机制详细参数
        self.coop_group = QGroupBox("协同机制参数 (A*大步长 + 斜圆柱体)")
        coop_layout = QGridLayout(self.coop_group)

        self.cb_use_global = QCheckBox("启用A*全局路径引导")
        self.cb_use_global.setChecked(True)
        coop_layout.addWidget(self.cb_use_global, 0, 0, 1, 2)

        coop_layout.addWidget(QLabel("大步长(米):"), 1, 0)
        self.spin_step_size = QDoubleSpinBox()
        self.spin_step_size.setRange(5, 100)
        self.spin_step_size.setValue(15.0)
        self.spin_step_size.setSingleStep(1.0)
        coop_layout.addWidget(self.spin_step_size, 1, 1)

        coop_layout.addWidget(QLabel("圆柱体半径(米):"), 2, 0)
        self.spin_cylinder_radius = QDoubleSpinBox()
        self.spin_cylinder_radius.setRange(5, 50)
        self.spin_cylinder_radius.setValue(20.0)
        self.spin_cylinder_radius.setSingleStep(1.0)
        coop_layout.addWidget(self.spin_cylinder_radius, 2, 1)

        coop_layout.addWidget(QLabel("切换距离阈值(米):"), 3, 0)
        self.spin_switch_thresh = QDoubleSpinBox()
        self.spin_switch_thresh.setRange(1, 20)
        self.spin_switch_thresh.setValue(5.0)
        coop_layout.addWidget(self.spin_switch_thresh, 3, 1)

        coop_layout.addWidget(QLabel("路径跟随奖励系数(偏离惩罚):"), 4, 0)
        self.spin_reward_follow = QDoubleSpinBox()
        self.spin_reward_follow.setRange(-10, 10)
        self.spin_reward_follow.setValue(-0.5)
        self.spin_reward_follow.setSingleStep(0.1)
        coop_layout.addWidget(self.spin_reward_follow, 4, 1)

        coop_layout.addWidget(QLabel("进度奖励系数:"), 5, 0)
        self.spin_reward_progress = QDoubleSpinBox()
        self.spin_reward_progress.setRange(0, 10)
        self.spin_reward_progress.setValue(1.0)
        coop_layout.addWidget(self.spin_reward_progress, 5, 1)

        coop_layout.addWidget(QLabel("超出圆柱惩罚:"), 6, 0)
        self.spin_out_penalty = QDoubleSpinBox()
        self.spin_out_penalty.setRange(-20, 0)
        self.spin_out_penalty.setValue(-5.0)
        coop_layout.addWidget(self.spin_out_penalty, 6, 1)

        alg_layout.addWidget(self.coop_group)
        self.cb_coop.toggled.connect(self.coop_group.setEnabled)
        self.coop_group.setEnabled(self.cb_coop.isChecked())

        layout.addWidget(group_alg)

        # 4. 配置操作按钮
        btn_config_layout = QHBoxLayout()
        btn_save_cfg = QPushButton("保存配置")
        btn_load_cfg = QPushButton("加载配置")
        btn_del_cfg = QPushButton("删除配置")
        btn_save_cfg.clicked.connect(self._on_save_config)
        btn_load_cfg.clicked.connect(self._on_load_config)
        btn_del_cfg.clicked.connect(self._on_delete_config)
        btn_config_layout.addWidget(btn_save_cfg)
        btn_config_layout.addWidget(btn_load_cfg)
        btn_config_layout.addWidget(btn_del_cfg)
        layout.addLayout(btn_config_layout)

        layout.addStretch()

    def _connect_signals(self):
        for edit in (self.edit_start_x, self.edit_start_y, self.edit_start_z,
                     self.edit_goal_x, self.edit_goal_y, self.edit_goal_z):
            edit.editingFinished.connect(self._update_start_goal_vis)

    # =========================================================
    # 对外接口
    # =========================================================
    def get_job_config(self) -> TrainJobConfig:
        """将当前所有控件值打包为 TrainJobConfig，供训练页调用"""
        return self._build_job_config()

    # =========================================================
    # 辅助方法
    # =========================================================
    def _update_start_goal_vis(self):
        if not hasattr(self.main_window, 'renderer'):
            return
        try:
            self.main_window.renderer.draw_start(
                (float(self.edit_start_x.text()),
                 float(self.edit_start_y.text()),
                 float(self.edit_start_z.text())),
                size=2.0, color='green'
            )
            self.main_window.renderer.draw_goal(
                (float(self.edit_goal_x.text()),
                 float(self.edit_goal_y.text()),
                 float(self.edit_goal_z.text())),
                size=2.0, color='red'
            )
        except Exception:
            pass

    def _render_mountain(self, clear_trajectory=True):
        t = self.current_mountain_data.terrain
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
        if clear_trajectory:
            self.main_window.renderer.clear_trajectory()
        self.main_window.render_terrain(X, Y, Z)
        for obs in self.current_mountain_data.obstacles:
            self.main_window.render_obstacle(obs)

    def _build_job_config(self) -> TrainJobConfig:
        uav_config = UAVConfig(
            start_x=float(self.edit_start_x.text()),
            start_y=float(self.edit_start_y.text()),
            start_z=float(self.edit_start_z.text()),
            goal_x=float(self.edit_goal_x.text()),
            goal_y=float(self.edit_goal_y.text()),
            goal_z=float(self.edit_goal_z.text()),
            max_speed=float(self.edit_max_speed.text()),
            max_accel=float(self.edit_max_accel.text()),
            min_altitude=float(self.edit_min_alt.text()),
            max_altitude=float(self.edit_max_alt.text()),
            min_altitude_ref=self.cb_min_alt_ref.currentData(),
            max_altitude_ref=self.cb_max_alt_ref.currentData(),
            altitude_ref="amsl"
        )
        algorithm_config = AlgorithmConfig(
            learning_rate=float(self.edit_lr.text()),
            gamma=float(self.edit_gamma.text()),
            batch_size=int(self.edit_batch.value()),
            hidden_size=self.edit_hidden_size.value(),
            buffer_capacity=self.edit_buffer_capacity.value(),
            reward_dist=float(self.edit_r_dist.text()),
            reward_collision=float(self.edit_r_collision.text()),
            reward_smooth=float(self.edit_r_smooth.text()),
            cooperative=self.cb_coop.isChecked(),
            use_global_path=self.cb_use_global.isChecked(),
            global_step_size=self.spin_step_size.value(),
            cylinder_radius=self.spin_cylinder_radius.value(),
            switch_threshold=self.spin_switch_thresh.value(),
            reward_path_follow=self.spin_reward_follow.value(),
            reward_path_progress=self.spin_reward_progress.value(),
            reward_out_of_cylinder=self.spin_out_penalty.value()
        )
        train_config = TrainConfig(
            num_epochs=1000,     # 默认值，训练页会覆盖
            save_interval=100,
            max_steps=500
        )
        return TrainJobConfig(
            mountain=self.current_mountain_data,
            uav=uav_config,
            algorithm=algorithm_config,
            training=train_config
        )

    # =========================================================
    # 山区加载
    # =========================================================
    def _on_load_mountain(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择山区文件夹", self.main_window.file_manager.get_root_dir())
        if not dir_path:
            return
        data_path = os.path.join(dir_path, "山区建模数据.json")
        if not os.path.exists(data_path):
            QMessageBox.warning(self, "警告", "所选文件夹中未找到山区建模数据.json")
            return
        try:
            data = load_json(data_path)
            self.current_mountain_data = MountainData.from_dict(data)
            self.current_mountain_name = os.path.basename(dir_path)
            self.label_mountain_info.setText(f"已加载: {self.current_mountain_name}")
            self._render_mountain(clear_trajectory=True)
            self._update_start_goal_vis()
            self.main_window.log(f"山区数据已加载: {self.current_mountain_name}")
            QMessageBox.information(self, "成功", f"山区 '{self.current_mountain_name}' 加载成功")
        except Exception as e:
            self.main_window.log(f"加载山区数据失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"加载山区数据失败：{str(e)}")

    # =========================================================
    # 起点/终点拾取
    # =========================================================
    def _on_pick_start(self):
        if not hasattr(self.main_window, 'renderer'):
            QMessageBox.warning(self, "警告", "渲染器未初始化")
            return
        self.main_window.log("请在3D视图中点击选择起点位置")
        self.btn_pick_start.setEnabled(False)
        self.btn_pick_start.setText("拾取中...")
        self.main_window.renderer.enable_pick_mode(self._on_start_picked)

    def _on_start_picked(self, point):
        self.edit_start_x.setText(f"{point[0]:.2f}")
        self.edit_start_y.setText(f"{point[1]:.2f}")
        self.edit_start_z.setText(f"{point[2]:.2f}")
        self.btn_pick_start.setEnabled(True)
        self.btn_pick_start.setText("拾取起点")
        self.main_window.log(f"已拾取起点: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})")
        self._update_start_goal_vis()

    def _on_pick_goal(self):
        if not hasattr(self.main_window, 'renderer'):
            QMessageBox.warning(self, "警告", "渲染器未初始化")
            return
        self.main_window.log("请在3D视图中点击选择终点位置")
        self.btn_pick_goal.setEnabled(False)
        self.btn_pick_goal.setText("拾取中...")
        self.main_window.renderer.enable_pick_mode(self._on_goal_picked)

    def _on_goal_picked(self, point):
        self.edit_goal_x.setText(f"{point[0]:.2f}")
        self.edit_goal_y.setText(f"{point[1]:.2f}")
        self.edit_goal_z.setText(f"{point[2]:.2f}")
        self.btn_pick_goal.setEnabled(True)
        self.btn_pick_goal.setText("拾取终点")
        self.main_window.log(f"已拾取终点: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})")
        self._update_start_goal_vis()

    # =========================================================
    # 配置保存/加载/删除
    # =========================================================
    def _on_save_config(self):
        if not self.current_mountain_name:
            QMessageBox.warning(self, "警告", "请先加载一个山区")
            return
        config_name, ok = QInputDialog.getText(
            self, "保存配置", "请输入配置名称（例如：无人机与算法_1）:",
            text=self.current_config_name or "无人机与算法_1")
        if not ok or not config_name.strip():
            return
        config_name = config_name.strip()
        file_path = self.main_window.file_manager.get_uav_alg_config_path(
            self.current_mountain_name, config_name)
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            job_config = self._build_job_config()
            save_json(job_config.to_dict(), file_path)
            self.current_config_name = config_name
            self.main_window.log(f"训练配置已保存至: {file_path}")
            QMessageBox.information(self, "成功", f"配置已保存到：{file_path}")
        except Exception as e:
            self.main_window.log(f"保存配置失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"保存配置失败：{str(e)}")

    def _on_load_config(self):
        if not self.current_mountain_name:
            QMessageBox.warning(self, "警告", "请先加载一个山区")
            return
        mountain_path = self.main_window.file_manager.get_mountain_path(self.current_mountain_name)
        if not os.path.exists(mountain_path):
            return
        config_dirs = [d for d in os.listdir(mountain_path)
                       if os.path.isdir(os.path.join(mountain_path, d)) and d.startswith("无人机与算法_")]
        if not config_dirs:
            QMessageBox.information(self, "提示", "该山区下没有找到任何配置文件夹")
            return
        config_name, ok = QInputDialog.getItem(
            self, "选择配置", "请选择一个配置:", config_dirs, 0, False)
        if not ok or not config_name:
            return
        file_path = self.main_window.file_manager.get_uav_alg_config_path(
            self.current_mountain_name, config_name)
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "警告", f"配置文件不存在：{file_path}")
            return
        try:
            data = load_json(file_path)
            job_config = TrainJobConfig.from_dict(data)

            u = job_config.uav
            self.edit_start_x.setText(str(u.start_x))
            self.edit_start_y.setText(str(u.start_y))
            self.edit_start_z.setText(str(u.start_z))
            self.edit_goal_x.setText(str(u.goal_x))
            self.edit_goal_y.setText(str(u.goal_y))
            self.edit_goal_z.setText(str(u.goal_z))
            self.edit_max_speed.setText(str(u.max_speed))
            self.edit_max_accel.setText(str(u.max_accel))
            self.edit_min_alt.setText(str(u.min_altitude))
            self.edit_max_alt.setText(str(u.max_altitude))
            idx = self.cb_min_alt_ref.findData(u.min_altitude_ref)
            if idx >= 0:
                self.cb_min_alt_ref.setCurrentIndex(idx)
            idx = self.cb_max_alt_ref.findData(u.max_altitude_ref)
            if idx >= 0:
                self.cb_max_alt_ref.setCurrentIndex(idx)

            a = job_config.algorithm
            self.edit_lr.setText(str(a.learning_rate))
            self.edit_gamma.setText(str(a.gamma))
            self.edit_batch.setValue(a.batch_size)
            self.edit_hidden_size.setValue(a.hidden_size)
            self.edit_buffer_capacity.setValue(a.buffer_capacity)
            self.edit_r_dist.setText(str(a.reward_dist))
            self.edit_r_collision.setText(str(a.reward_collision))
            self.edit_r_smooth.setText(str(a.reward_smooth))
            self.cb_coop.setChecked(a.cooperative)
            self.cb_use_global.setChecked(getattr(a, 'use_global_path', True))
            self.spin_step_size.setValue(getattr(a, 'global_step_size', 15.0))
            self.spin_cylinder_radius.setValue(getattr(a, 'cylinder_radius', 20.0))
            self.spin_switch_thresh.setValue(getattr(a, 'switch_threshold', 5.0))
            self.spin_reward_follow.setValue(getattr(a, 'reward_path_follow', -0.5))
            self.spin_reward_progress.setValue(getattr(a, 'reward_path_progress', 1.0))
            self.spin_out_penalty.setValue(getattr(a, 'reward_out_of_cylinder', -5.0))

            self.current_mountain_data = job_config.mountain
            self.current_mountain_name = os.path.basename(
                os.path.dirname(os.path.dirname(file_path)))
            self.current_config_name = config_name

            self.label_mountain_info.setText(f"已加载: {self.current_mountain_name}")
            self._render_mountain(clear_trajectory=True)
            self._update_start_goal_vis()

            self.main_window.log(f"训练配置已加载: {file_path}")
            QMessageBox.information(self, "成功", f"配置 '{config_name}' 加载成功")
        except Exception as e:
            self.main_window.log(f"加载配置失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"加载配置失败：{str(e)}")

    def _on_delete_config(self):
        if not self.current_mountain_name:
            QMessageBox.warning(self, "警告", "请先加载一个山区")
            return
        mountain_path = self.main_window.file_manager.get_mountain_path(self.current_mountain_name)
        if not os.path.exists(mountain_path):
            return
        config_dirs = [d for d in os.listdir(mountain_path)
                       if os.path.isdir(os.path.join(mountain_path, d)) and d.startswith("无人机与算法_")]
        if not config_dirs:
            QMessageBox.information(self, "提示", "没有可删除的配置")
            return
        config_name, ok = QInputDialog.getItem(
            self, "选择要删除的配置", "请选择一个配置:", config_dirs, 0, False)
        if not ok or not config_name:
            return
        file_path = self.main_window.file_manager.get_uav_alg_config_path(
            self.current_mountain_name, config_name)
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除配置 '{config_name}' 吗？\n文件：{file_path}",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    dir_p = os.path.dirname(file_path)
                    if os.path.exists(dir_p) and not os.listdir(dir_p):
                        os.rmdir(dir_p)
                    self.main_window.log(f"已删除配置: {file_path}")
                    if self.current_config_name == config_name:
                        self.current_config_name = None
                    QMessageBox.information(self, "成功", "配置已删除")
                else:
                    QMessageBox.warning(self, "警告", "文件不存在")
            except Exception as e:
                self.main_window.log(f"删除配置失败: {str(e)}")
                QMessageBox.critical(self, "错误", f"删除失败：{str(e)}")
