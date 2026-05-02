"""
训练配置标签页：包含山区加载、无人机/算法参数、训练控制UI，并绑定实际训练逻辑
优化版：独立显示和设置经验保存目录，支持训练中动态切换保存路径，简化操作流程
新增：全局-局部协同机制（A*大步长路径 + 斜圆柱体约束）的配置界面
"""

import os
import re
import pickle
from collections import deque
from pathlib import Path
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QLineEdit, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QInputDialog, QFormLayout, QGridLayout
)
from PyQt5.QtCore import Qt, QSettings

from data.models import (
    TerrainData, MountainData, UAVConfig,
    AlgorithmConfig, TrainConfig, TrainJobConfig
)
from business.train_controller import TrainController
from business.file_manager import FileManager
from data.serializer import load_json, save_json
from business.terrain_builder import generate_terrain, generate_terrain_perlin


class TabTrain(QWidget):
    """训练配置标签页"""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.current_mountain_data = MountainData()
        self.current_mountain_name = None
        self.current_config_name = None
        self.train_controller = None
        self.save_dir = None                     # 保存目录（Path对象）
        self.epoch_stats_list = []               # 用于存储本轮训练所有轮次的统计信息

        self.init_ui()
        self._connect_signals()
        self._init_save_dir()                    # 初始化保存目录

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

        # 最低高度
        constraint_layout.addWidget(QLabel("最低高度:"))
        self.edit_min_alt = QLineEdit("5")
        constraint_layout.addWidget(self.edit_min_alt)
        self.cb_min_alt_ref = QComboBox()
        self.cb_min_alt_ref.addItem("海拔", "amsl")
        self.cb_min_alt_ref.addItem("离地", "agl")
        self.cb_min_alt_ref.setCurrentIndex(0)
        constraint_layout.addWidget(self.cb_min_alt_ref)

        # 最高高度
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

        # 隐藏层大小和缓冲区容量
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

        # 奖励系数
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

        # ========== 全局-局部协同机制详细参数 ==========
        self.coop_group = QGroupBox("协同机制参数 (A*大步长 + 斜圆柱体)")
        coop_layout = QGridLayout(self.coop_group)

        # 启用开关（与上面的复选框同步）
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

        # 根据协同总开关控制详细参数的启用状态
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

        # 5. 训练设置组
        group_train = QGroupBox("训练设置")
        train_layout = QVBoxLayout(group_train)

        # 显示当前加载信息
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel("当前山区:"))
        self.label_current_mountain = QLabel("未加载")
        info_layout.addWidget(self.label_current_mountain)
        info_layout.addWidget(QLabel("   当前算法:"))
        self.label_current_algorithm = QLabel("未加载")
        info_layout.addWidget(self.label_current_algorithm)
        info_layout.addStretch()
        train_layout.addLayout(info_layout)

        # 训练参数设置
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("训练轮数:"))
        self.edit_epochs = QSpinBox()
        self.edit_epochs.setRange(1, 10000)
        self.edit_epochs.setValue(1000)
        param_layout.addWidget(self.edit_epochs)

        param_layout.addWidget(QLabel("保存间隔(轮):"))
        self.edit_save_interval = QSpinBox()
        self.edit_save_interval.setRange(1, 1000)
        self.edit_save_interval.setValue(100)
        param_layout.addWidget(self.edit_save_interval)

        param_layout.addWidget(QLabel("最大步数:"))
        self.edit_max_steps = QSpinBox()
        self.edit_max_steps.setRange(1, 10000)
        self.edit_max_steps.setValue(500)
        param_layout.addWidget(self.edit_max_steps)

        train_layout.addLayout(param_layout)

        # 可视化设置
        vis_layout = QHBoxLayout()
        vis_layout.addWidget(QLabel("起点标记大小:"))
        self.edit_start_size = QDoubleSpinBox()
        self.edit_start_size.setRange(0.5, 20.0)
        self.edit_start_size.setValue(2.0)
        self.edit_start_size.setSingleStep(0.5)
        vis_layout.addWidget(self.edit_start_size)

        vis_layout.addWidget(QLabel("终点标记大小:"))
        self.edit_goal_size = QDoubleSpinBox()
        self.edit_goal_size.setRange(0.5, 20.0)
        self.edit_goal_size.setValue(2.0)
        self.edit_goal_size.setSingleStep(0.5)
        vis_layout.addWidget(self.edit_goal_size)

        vis_layout.addWidget(QLabel("无人机大小:"))
        self.edit_uav_size = QDoubleSpinBox()
        self.edit_uav_size.setRange(0.5, 20.0)
        self.edit_uav_size.setValue(2.0)
        self.edit_uav_size.setSingleStep(0.5)
        vis_layout.addWidget(self.edit_uav_size)

        self.cb_visualize = QCheckBox("训练时可视化轨迹")
        self.cb_visualize.setChecked(True)
        vis_layout.addWidget(self.cb_visualize)

        train_layout.addLayout(vis_layout)

        # ===== 经验保存目录设置 =====
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("经验保存目录:"))
        self.lbl_exp_dir = QLabel("未设置")
        self.lbl_exp_dir.setMinimumWidth(300)
        dir_layout.addWidget(self.lbl_exp_dir)
        self.btn_set_exp_dir = QPushButton("设置保存目录")
        self.btn_set_exp_dir.clicked.connect(self._on_set_exp_dir)
        dir_layout.addWidget(self.btn_set_exp_dir)
        train_layout.addLayout(dir_layout)

        # 经验文件路径显示
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("经验文件路径:"))
        self.lbl_exp_path = QLabel("")
        path_layout.addWidget(self.lbl_exp_path)
        path_layout.addStretch()
        train_layout.addLayout(path_layout)

        # ===== 本轮训练统计面板 =====
        self.stats_group = QGroupBox("本轮训练统计")
        stats_layout = QGridLayout(self.stats_group)

        self.lbl_success = QLabel("-")
        self.lbl_collision = QLabel("-")
        self.lbl_final_dist = QLabel("-")
        self.lbl_path_length = QLabel("-")
        self.lbl_avg_acc = QLabel("-")
        self.lbl_acc_var = QLabel("-")
        self.lbl_total_reward = QLabel("-")
        self.lbl_steps = QLabel("-")
        self.lbl_loss = QLabel("-")
        self.lbl_reward_dist = QLabel("-")
        self.lbl_reward_smooth = QLabel("-")
        self.lbl_reward_collision = QLabel("-")
        self.lbl_reward_goal = QLabel("-")

        items = [
            ("是否成功:", self.lbl_success),
            ("发生碰撞:", self.lbl_collision),
            ("终点距离 (m):", self.lbl_final_dist),
            ("路径长度 (m):", self.lbl_path_length),
            ("平均加速度 (m/s²):", self.lbl_avg_acc),
            ("加速度方差 (平滑度):", self.lbl_acc_var),
            ("累计奖励:", self.lbl_total_reward),
            ("步数:", self.lbl_steps),
            ("Actor损失:", self.lbl_loss),
            ("距离奖励:", self.lbl_reward_dist),
            ("平滑惩罚:", self.lbl_reward_smooth),
            ("碰撞惩罚:", self.lbl_reward_collision),
            ("终点奖励:", self.lbl_reward_goal),
        ]
        cols = 3
        for idx, (label_text, value_label) in enumerate(items):
            row = idx // cols
            col = idx % cols
            desc_label = QLabel(label_text)
            stats_layout.addWidget(desc_label, row, col*2)
            stats_layout.addWidget(value_label, row, col*2 + 1)

        train_layout.addWidget(self.stats_group)

        # ===== 本次训练汇总面板 =====
        self.summary_group = QGroupBox("本次训练汇总")
        summary_layout = QGridLayout(self.summary_group)

        self.lbl_avg_success = QLabel("-")
        self.lbl_avg_reward = QLabel("-")
        self.lbl_collision_rate = QLabel("-")
        self.lbl_avg_path_length = QLabel("-")
        self.lbl_avg_smoothness = QLabel("-")
        self.lbl_total_epochs = QLabel("-")
        self.lbl_avg_reward_dist = QLabel("-")
        self.lbl_avg_reward_smooth = QLabel("-")
        self.lbl_avg_reward_collision = QLabel("-")
        self.lbl_avg_reward_goal = QLabel("-")

        summary_items = [
            ("平均成功率:", self.lbl_avg_success),
            ("平均奖励:", self.lbl_avg_reward),
            ("碰撞率:", self.lbl_collision_rate),
            ("平均路径长度 (m):", self.lbl_avg_path_length),
            ("平均平滑度:", self.lbl_avg_smoothness),
            ("总轮次:", self.lbl_total_epochs),
            ("平均距离奖励:", self.lbl_avg_reward_dist),
            ("平均平滑惩罚:", self.lbl_avg_reward_smooth),
            ("平均碰撞惩罚:", self.lbl_avg_reward_collision),
            ("平均终点奖励:", self.lbl_avg_reward_goal),
        ]
        cols_summary = 2
        for idx, (label_text, value_label) in enumerate(summary_items):
            row = idx // cols_summary
            col = idx % cols_summary
            desc_label = QLabel(label_text)
            summary_layout.addWidget(desc_label, row, col*2)
            summary_layout.addWidget(value_label, row, col*2 + 1)

        train_layout.addWidget(self.summary_group)

        # 训练控制按钮
        btn_train_layout = QHBoxLayout()
        self.btn_start = QPushButton("开始训练")
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        self.btn_start.clicked.connect(self._on_start_training)

        self.btn_pause = QPushButton("暂停")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._on_pause_training)

        self.btn_resume = QPushButton("恢复")
        self.btn_resume.setEnabled(False)
        self.btn_resume.clicked.connect(self._on_resume_training)

        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop_training)

        self.btn_save_exp = QPushButton("保存当前经验")
        self.btn_save_exp.setEnabled(False)
        self.btn_save_exp.clicked.connect(self._on_save_experience)

        self.btn_load_exp = QPushButton("加载经验文件")
        self.btn_load_exp.setEnabled(False)
        self.btn_load_exp.clicked.connect(self._on_load_experience)

        self.btn_continuous_train = QPushButton("持续训练直到终点")
        self.btn_continuous_train.setEnabled(False)
        self.btn_continuous_train.clicked.connect(self._on_continuous_train)

        btn_train_layout.addWidget(self.btn_start)
        btn_train_layout.addWidget(self.btn_pause)
        btn_train_layout.addWidget(self.btn_resume)
        btn_train_layout.addWidget(self.btn_stop)
        btn_train_layout.addWidget(self.btn_save_exp)
        btn_train_layout.addWidget(self.btn_load_exp)
        btn_train_layout.addWidget(self.btn_continuous_train)
        train_layout.addLayout(btn_train_layout)

        layout.addWidget(group_train)
        layout.addStretch()

        self._update_info_labels()

    def _connect_signals(self):
        self.edit_start_x.editingFinished.connect(self._update_start_goal_vis)
        self.edit_start_y.editingFinished.connect(self._update_start_goal_vis)
        self.edit_start_z.editingFinished.connect(self._update_start_goal_vis)
        self.edit_goal_x.editingFinished.connect(self._update_start_goal_vis)
        self.edit_goal_y.editingFinished.connect(self._update_start_goal_vis)
        self.edit_goal_z.editingFinished.connect(self._update_start_goal_vis)
        self.edit_start_size.valueChanged.connect(self._update_start_goal_vis)
        self.edit_goal_size.valueChanged.connect(self._update_start_goal_vis)

    # ---------- 保存目录管理 ----------
    def _init_save_dir(self):
        if self.current_config_name and self.current_mountain_name:
            config_dir = self.main_window.file_manager.get_uav_alg_dir_path(
                self.current_mountain_name, self.current_config_name
            )
            self.save_dir = Path(config_dir).resolve()
        else:
            settings = QSettings("UAVLab", "UAVSimulator")
            last_dir = settings.value("last_exp_dir", "")
            if last_dir and Path(last_dir).exists():
                self.save_dir = Path(last_dir)
            else:
                desktop = Path.home() / "Desktop"
                default_dir = desktop / "算法训练集" / "经验保存"
                self.save_dir = default_dir
        self._update_exp_dir_display()

    def _update_exp_dir_display(self):
        if self.save_dir:
            self.lbl_exp_dir.setText(str(self.save_dir))
            buffer_path = self.save_dir / "model.buffer.pkl"
            self.lbl_exp_path.setText(str(buffer_path))
            if buffer_path.exists():
                size = buffer_path.stat().st_size
                self.lbl_exp_path.setToolTip(f"文件大小: {size} 字节")
            else:
                self.lbl_exp_path.setToolTip("文件不存在，训练时将自动创建")
        else:
            self.lbl_exp_dir.setText("未设置")
            self.lbl_exp_path.setText("")

    def _on_set_exp_dir(self):
        start_dir = str(self.save_dir) if self.save_dir else ""
        dir_path = QFileDialog.getExistingDirectory(self, "选择经验保存目录", start_dir)
        if not dir_path:
            return
        new_dir = Path(dir_path).resolve()
        if self.train_controller and self.train_controller.is_training:
            reply = QMessageBox.question(
                self, "训练中切换目录",
                "训练正在运行，是否将当前经验保存到新目录？\n否则当前内存中的经验不会写入新目录。",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Cancel:
                return
            if reply == QMessageBox.Yes:
                self.train_controller.change_save_dir(new_dir, save_current=True)
            else:
                self.train_controller.change_save_dir(new_dir, save_current=False)
        else:
            self.save_dir = new_dir
            QSettings("UAVLab", "UAVSimulator").setValue("last_exp_dir", str(new_dir))
            self._update_exp_dir_display()
            if self.current_config_name:
                self.btn_load_exp.setEnabled(True)
                self.btn_continuous_train.setEnabled(True)

    # ---------- 辅助方法 ----------
    def _update_start_goal_vis(self):
        if not hasattr(self.main_window, 'renderer'):
            return
        try:
            start_x = float(self.edit_start_x.text())
            start_y = float(self.edit_start_y.text())
            start_z = float(self.edit_start_z.text())
            goal_x = float(self.edit_goal_x.text())
            goal_y = float(self.edit_goal_y.text())
            goal_z = float(self.edit_goal_z.text())
            start_size = self.edit_start_size.value()
            goal_size = self.edit_goal_size.value()
            self.main_window.renderer.draw_start((start_x, start_y, start_z), size=start_size, color='green')
            self.main_window.renderer.draw_goal((goal_x, goal_y, goal_z), size=goal_size, color='red')
        except:
            pass

    def _update_info_labels(self):
        mountain_text = self.current_mountain_name if self.current_mountain_name else "未加载"
        algo_text = self.current_config_name if self.current_config_name else "未加载"
        self.label_current_mountain.setText(mountain_text)
        self.label_current_algorithm.setText(algo_text)

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

    # ---------- 山区加载 ----------
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
            self._update_info_labels()
            self._render_mountain(clear_trajectory=True)
            self._update_start_goal_vis()
            self.main_window.log(f"山区数据已加载: {self.current_mountain_name}")
            QMessageBox.information(self, "成功", f"山区 '{self.current_mountain_name}' 加载成功")
        except Exception as e:
            self.main_window.log(f"加载山区数据失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"加载山区数据失败：{str(e)}")

    # ---------- 起点/终点拾取 ----------
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

    # ---------- 构建训练任务配置 ----------
    def _build_job_config(self):
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
            # 全局-局部协同参数
            use_global_path=self.cb_use_global.isChecked(),
            global_step_size=self.spin_step_size.value(),
            cylinder_radius=self.spin_cylinder_radius.value(),
            switch_threshold=self.spin_switch_thresh.value(),
            reward_path_follow=self.spin_reward_follow.value(),
            reward_path_progress=self.spin_reward_progress.value(),
            reward_out_of_cylinder=self.spin_out_penalty.value()
        )
        train_config = TrainConfig(
            num_epochs=int(self.edit_epochs.value()),
            save_interval=int(self.edit_save_interval.value()),
            max_steps=self.edit_max_steps.value()
        )
        return TrainJobConfig(
            mountain=self.current_mountain_data,
            uav=uav_config,
            algorithm=algorithm_config,
            training=train_config
        )

    # ---------- 配置保存/加载 ----------
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
            self._update_info_labels()
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

            # 更新UI
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
            index = self.cb_min_alt_ref.findData(u.min_altitude_ref)
            if index >= 0:
                self.cb_min_alt_ref.setCurrentIndex(index)
            index = self.cb_max_alt_ref.findData(u.max_altitude_ref)
            if index >= 0:
                self.cb_max_alt_ref.setCurrentIndex(index)

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
            # 协同参数
            self.cb_use_global.setChecked(getattr(a, 'use_global_path', True))
            self.spin_step_size.setValue(getattr(a, 'global_step_size', 15.0))
            self.spin_cylinder_radius.setValue(getattr(a, 'cylinder_radius', 20.0))
            self.spin_switch_thresh.setValue(getattr(a, 'switch_threshold', 5.0))
            self.spin_reward_follow.setValue(getattr(a, 'reward_path_follow', -0.5))
            self.spin_reward_progress.setValue(getattr(a, 'reward_path_progress', 1.0))
            self.spin_out_penalty.setValue(getattr(a, 'reward_out_of_cylinder', -5.0))

            t = job_config.training
            self.edit_epochs.setValue(t.num_epochs)
            self.edit_save_interval.setValue(t.save_interval)
            self.edit_max_steps.setValue(t.max_steps)

            self.current_mountain_data = job_config.mountain
            self.current_mountain_name = os.path.basename(
                os.path.dirname(os.path.dirname(file_path)))
            self.current_config_name = config_name

            self._update_info_labels()
            self.label_mountain_info.setText(f"已加载: {self.current_mountain_name}")
            self._render_mountain(clear_trajectory=True)
            self._update_start_goal_vis()

            # 设置保存目录为配置文件夹
            config_dir = self.main_window.file_manager.get_uav_alg_dir_path(
                self.current_mountain_name, self.current_config_name
            )
            self.save_dir = Path(config_dir).resolve()
            self._update_exp_dir_display()

            self.btn_load_exp.setEnabled(True)
            self.btn_continuous_train.setEnabled(True)
            self.btn_save_exp.setEnabled(False)

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
                    dir_path = os.path.dirname(file_path)
                    if os.path.exists(dir_path) and not os.listdir(dir_path):
                        os.rmdir(dir_path)
                    self.main_window.log(f"已删除配置: {file_path}")
                    if self.current_config_name == config_name:
                        self.current_config_name = None
                        self._update_info_labels()
                    QMessageBox.information(self, "成功", "配置已删除")
                else:
                    QMessageBox.warning(self, "警告", "文件不存在")
            except Exception as e:
                self.main_window.log(f"删除配置失败: {str(e)}")
                QMessageBox.critical(self, "错误", f"删除失败：{str(e)}")

    # ---------- 经验文件操作 ----------
    def _on_load_experience(self):
        if self.train_controller and self.train_controller.is_training:
            QMessageBox.warning(self, "警告", "训练中无法加载经验，请先停止训练")
            return
        if not self.save_dir:
            QMessageBox.warning(self, "警告", "请先设置经验保存目录")
            return
        file_path, ok = QFileDialog.getOpenFileName(
            self, "选择经验文件", str(self.save_dir),
            "Pickle Files (*.pkl);;All Files (*)"
        )
        if not ok or not file_path:
            return
        target_path = self.save_dir / "model.buffer.pkl"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(file_path, 'rb') as f:
                loaded_buffer = pickle.load(f)
            buffer_capacity = self.edit_buffer_capacity.value()
            if hasattr(loaded_buffer, 'maxlen') and loaded_buffer.maxlen != buffer_capacity:
                reply = QMessageBox.question(
                    self, "容量不匹配",
                    f"加载的经验文件容量为 {loaded_buffer.maxlen}，当前配置容量为 {buffer_capacity}。\n"
                    "是否继续？经验可能被截断或浪费空间。",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return
            with open(target_path, 'wb') as f:
                pickle.dump(loaded_buffer, f)
            self.main_window.log(f"经验文件已加载到: {target_path}, 包含 {len(loaded_buffer)} 条经验")
            QMessageBox.information(self, "成功", f"已加载经验文件，包含 {len(loaded_buffer)} 条经验。\n经验将保存在：{target_path}")
        except Exception as e:
            self.main_window.log(f"加载经验文件失败: {e}")
            QMessageBox.critical(self, "错误", f"加载失败：{e}")

    def _on_save_experience(self):
        if self.train_controller and self.train_controller.is_training:
            self.train_controller.save_experience_only()
        else:
            QMessageBox.information(self, "提示", "请先开始训练，训练中可保存当前经验")

    # ---------- 持续训练直到终点 ----------
    def _on_continuous_train(self):
        if not self.current_mountain_name:
            reply = QMessageBox.warning(
                self,
                "未加载山区",
                "当前未加载任何山区数据，将使用默认山区进行训练。\n是否继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
            self.current_mountain_name = "默认山区"
            self.current_mountain_data = MountainData()
            self._update_info_labels()
        if not self.current_config_name:
            self.current_config_name = "无人机与算法_1"
        run_index = self._get_next_run_index(self.current_mountain_name, self.current_config_name)
        job_config = self._build_job_config()
        self.train_controller = TrainController(
            main_window=self.main_window,
            file_manager=self.main_window.file_manager,
            job_config=job_config,
            mountain_name=self.current_mountain_name,
            config_name=self.current_config_name,
            run_index=run_index,
            save_dir=self.save_dir
        )
        self.train_controller.training_started.connect(self._on_training_started)
        self.train_controller.training_paused.connect(self._on_training_paused)
        self.train_controller.training_stopped.connect(self._on_training_stopped)
        self.train_controller.epoch_completed.connect(self._on_epoch_completed)
        self.train_controller.trajectory_generated.connect(self._on_trajectory_generated)

        self.epoch_stats_list = []
        self.lbl_avg_success.setText("-")
        self.lbl_avg_reward.setText("-")
        self.lbl_collision_rate.setText("-")
        self.lbl_avg_path_length.setText("-")
        self.lbl_avg_smoothness.setText("-")
        self.lbl_total_epochs.setText("-")
        self.lbl_avg_reward_dist.setText("-")
        self.lbl_avg_reward_smooth.setText("-")
        self.lbl_avg_reward_collision.setText("-")
        self.lbl_avg_reward_goal.setText("-")

        self.train_controller.continuous_mode = True
        self.train_controller.start_training()

    # ---------- 训练控制 ----------
    def _get_next_run_index(self, mountain_name, config_name):
        base_dir = self.main_window.file_manager.get_uav_alg_dir_path(mountain_name, config_name)
        if not os.path.exists(base_dir):
            return 1
        max_idx = 0
        pattern = re.compile(r"第(\d+)次训练过程")
        for name in os.listdir(base_dir):
            m = pattern.match(name)
            if m:
                idx = int(m.group(1))
                max_idx = max(max_idx, idx)
        return max_idx + 1

    def _on_start_training(self):
        if not self.current_mountain_name:
            reply = QMessageBox.warning(
                self,
                "未加载山区",
                "当前未加载任何山区数据，将使用默认山区进行训练。\n是否继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
            self.current_mountain_name = "默认山区"
            self.current_mountain_data = MountainData()
            self._update_info_labels()
        if not self.current_config_name:
            self.current_config_name = "无人机与算法_1"
        run_index = self._get_next_run_index(self.current_mountain_name, self.current_config_name)
        job_config = self._build_job_config()
        self.train_controller = TrainController(
            main_window=self.main_window,
            file_manager=self.main_window.file_manager,
            job_config=job_config,
            mountain_name=self.current_mountain_name,
            config_name=self.current_config_name,
            run_index=run_index,
            save_dir=self.save_dir
        )
        self.train_controller.training_started.connect(self._on_training_started)
        self.train_controller.training_paused.connect(self._on_training_paused)
        self.train_controller.training_stopped.connect(self._on_training_stopped)
        self.train_controller.epoch_completed.connect(self._on_epoch_completed)
        self.train_controller.trajectory_generated.connect(self._on_trajectory_generated)

        self.epoch_stats_list = []
        self.lbl_avg_success.setText("-")
        self.lbl_avg_reward.setText("-")
        self.lbl_collision_rate.setText("-")
        self.lbl_avg_path_length.setText("-")
        self.lbl_avg_smoothness.setText("-")
        self.lbl_total_epochs.setText("-")
        self.lbl_avg_reward_dist.setText("-")
        self.lbl_avg_reward_smooth.setText("-")
        self.lbl_avg_reward_collision.setText("-")
        self.lbl_avg_reward_goal.setText("-")

        self.train_controller.continuous_mode = False
        self.train_controller.start_training()

    def _on_training_started(self):
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_resume.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_save_exp.setEnabled(True)
        self.btn_load_exp.setEnabled(False)
        self.btn_continuous_train.setEnabled(False)
        self.main_window.log("【训练】训练已开始")

    def _on_pause_training(self):
        if self.train_controller:
            self.train_controller.pause_training()

    def _on_training_paused(self):
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(True)
        self.main_window.log("【训练】训练已暂停")

    def _on_resume_training(self):
        if self.train_controller:
            self.train_controller.resume_training()
            self.btn_pause.setEnabled(True)
            self.btn_resume.setEnabled(False)

    def _on_stop_training(self):
        if self.train_controller:
            self.train_controller.stop_training()

    def _on_training_stopped(self):
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_stop.setEnabled(False)
        if self.current_config_name:
            self.btn_save_exp.setEnabled(False)
            self.btn_load_exp.setEnabled(True)
            self.btn_continuous_train.setEnabled(True)
        else:
            self.btn_save_exp.setEnabled(False)
            self.btn_load_exp.setEnabled(False)
            self.btn_continuous_train.setEnabled(False)
        self.train_controller = None
        self.main_window.log("【训练】训练已停止")

        self.lbl_success.setText("-")
        self.lbl_collision.setText("-")
        self.lbl_final_dist.setText("-")
        self.lbl_path_length.setText("-")
        self.lbl_avg_acc.setText("-")
        self.lbl_acc_var.setText("-")
        self.lbl_total_reward.setText("-")
        self.lbl_steps.setText("-")
        self.lbl_loss.setText("-")
        self.lbl_reward_dist.setText("-")
        self.lbl_reward_smooth.setText("-")
        self.lbl_reward_collision.setText("-")
        self.lbl_reward_goal.setText("-")

        if self.epoch_stats_list:
            self._show_training_summary()

    def _on_epoch_completed(self, epoch, stats):
        success = stats.get('success', False)
        collision = stats.get('collision', False)
        total_reward = stats.get('total_reward', 0.0)
        final_distance = stats.get('final_distance', 0.0)
        path_length = stats.get('path_length', 0.0)
        avg_acc = stats.get('avg_acceleration', 0.0)
        acc_var = stats.get('acceleration_variance', 0.0)
        steps = stats.get('steps', 0)
        loss = stats.get('loss', 0.0)
        reason = stats.get('termination_reason', '未知')
        reward_dist = stats.get('reward_dist', 0.0)
        reward_smooth = stats.get('reward_smooth', 0.0)
        reward_collision = stats.get('reward_collision', 0.0)
        reward_goal = stats.get('reward_goal', 0.0)

        self.main_window.log(f"【训练】第{epoch}轮: 成功={success}, 碰撞={collision}, 奖励={total_reward:.2f}, 终止原因={reason}")

        self.lbl_success.setText("是" if success else "否")
        self.lbl_collision.setText("是" if collision else "否")
        self.lbl_final_dist.setText(f"{final_distance:.2f}")
        self.lbl_path_length.setText(f"{path_length:.2f}")
        self.lbl_avg_acc.setText(f"{avg_acc:.3f}")
        self.lbl_acc_var.setText(f"{acc_var:.3f}")
        self.lbl_total_reward.setText(f"{total_reward:.2f}")
        self.lbl_steps.setText(f"{steps}")
        self.lbl_loss.setText(f"{loss:.6f}")
        self.lbl_reward_dist.setText(f"{reward_dist:.2f}")
        self.lbl_reward_smooth.setText(f"{reward_smooth:.2f}")
        self.lbl_reward_collision.setText(f"{reward_collision:.2f}")
        self.lbl_reward_goal.setText(f"{reward_goal:.2f}")

        self.epoch_stats_list.append(stats)

    def _on_trajectory_generated(self, trajectory):
        if self.cb_visualize.isChecked():
            self.main_window.renderer.draw_trajectory(trajectory, color='orange', line_width=2)
            self.main_window.log("【训练】轨迹已更新至3D画布")
        else:
            self.main_window.log("【训练】轨迹已生成（可视化已关闭）")

    def _show_training_summary(self):
        if not self.epoch_stats_list:
            return
        n = len(self.epoch_stats_list)
        success_sum = sum(1 for s in self.epoch_stats_list if s.get('success', False))
        avg_success = success_sum / n
        avg_reward = np.mean([s.get('total_reward', 0.0) for s in self.epoch_stats_list])
        collision_sum = sum(1 for s in self.epoch_stats_list if s.get('collision', False))
        collision_rate = collision_sum / n
        avg_path_length = np.mean([s.get('path_length', 0.0) for s in self.epoch_stats_list])
        avg_smoothness = np.mean([s.get('acceleration_variance', 0.0) for s in self.epoch_stats_list])
        avg_reward_dist = np.mean([s.get('reward_dist', 0.0) for s in self.epoch_stats_list])
        avg_reward_smooth = np.mean([s.get('reward_smooth', 0.0) for s in self.epoch_stats_list])
        avg_reward_collision = np.mean([s.get('reward_collision', 0.0) for s in self.epoch_stats_list])
        avg_reward_goal = np.mean([s.get('reward_goal', 0.0) for s in self.epoch_stats_list])

        self.lbl_avg_success.setText(f"{avg_success:.2%}")
        self.lbl_avg_reward.setText(f"{avg_reward:.2f}")
        self.lbl_collision_rate.setText(f"{collision_rate:.2%}")
        self.lbl_avg_path_length.setText(f"{avg_path_length:.2f}")
        self.lbl_avg_smoothness.setText(f"{avg_smoothness:.4f}")
        self.lbl_total_epochs.setText(str(n))
        self.lbl_avg_reward_dist.setText(f"{avg_reward_dist:.2f}")
        self.lbl_avg_reward_smooth.setText(f"{avg_reward_smooth:.2f}")
        self.lbl_avg_reward_collision.setText(f"{avg_reward_collision:.2f}")
        self.lbl_avg_reward_goal.setText(f"{avg_reward_goal:.2f}")

        self.main_window.log(f"【训练汇总】共{n}轮，成功率={avg_success:.2%}，碰撞率={collision_rate:.2%}，平均奖励={avg_reward:.2f}")