"""
训练页：加载山区/算法配置（快捷入口）、训练参数、可视化设置、
经验目录管理、实时统计面板、训练控制按钮及全部训练回调逻辑。
依赖 TabConfig 实例提供 get_job_config()。
"""

import os
import re
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QDoubleSpinBox, QCheckBox, QSpinBox,
    QFileDialog, QMessageBox, QGridLayout
)
from PyQt5.QtCore import QSettings

from data.models import MountainData, TrainJobConfig
from data.serializer import load_json
from business.train_controller import TrainController
from business.terrain_builder import generate_terrain, generate_terrain_perlin


class TabRun(QWidget):
    """训练页：运行训练、实时统计、经验管理"""

    def __init__(self, main_window, config_tab):
        """
        :param main_window: 主窗口引用
        :param config_tab: TabConfig 实例，用于获取参数配置
        """
        super().__init__()
        self.main_window = main_window
        self.config_tab = config_tab          # 配置页引用

        self.train_controller = None
        self.save_dir = None                  # 经验保存目录（Path）
        self.epoch_stats_list = []            # 本轮训练所有轮次统计

        self.init_ui()
        self._connect_signals()
        self._init_save_dir()

    # =========================================================
    # UI 构建
    # =========================================================
    def init_ui(self):
        layout = QVBoxLayout(self)

        group_run = QGroupBox("训练设置")
        run_layout = QVBoxLayout(group_run)

        # ----- 快速加载：山区 -----
        load_mountain_layout = QHBoxLayout()
        self.btn_load_mountain = QPushButton("加载山区")
        self.btn_load_mountain.clicked.connect(self._on_load_mountain)
        load_mountain_layout.addWidget(self.btn_load_mountain)
        self.lbl_mountain_info = QLabel("未加载山区数据")
        load_mountain_layout.addWidget(self.lbl_mountain_info)
        load_mountain_layout.addStretch()
        run_layout.addLayout(load_mountain_layout)

        # ----- 快速加载：算法配置 -----
        load_config_layout = QHBoxLayout()
        self.btn_load_config = QPushButton("加载算法配置")
        self.btn_load_config.clicked.connect(self._on_load_config)
        load_config_layout.addWidget(self.btn_load_config)
        self.lbl_config_info = QLabel("未加载算法配置")
        load_config_layout.addWidget(self.lbl_config_info)
        load_config_layout.addStretch()
        run_layout.addLayout(load_config_layout)

        # ----- 当前加载状态汇总 -----
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel("当前山区:"))
        self.label_current_mountain = QLabel("未加载")
        info_layout.addWidget(self.label_current_mountain)
        info_layout.addWidget(QLabel("   当前算法:"))
        self.label_current_algorithm = QLabel("未加载")
        info_layout.addWidget(self.label_current_algorithm)
        info_layout.addStretch()
        run_layout.addLayout(info_layout)

        # ----- 训练参数 -----
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
        run_layout.addLayout(param_layout)

        # ----- 可视化设置 -----
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
        run_layout.addLayout(vis_layout)

        # ----- 经验保存目录 -----
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("经验保存目录:"))
        self.lbl_exp_dir = QLabel("未设置")
        self.lbl_exp_dir.setMinimumWidth(300)
        dir_layout.addWidget(self.lbl_exp_dir)
        self.btn_set_exp_dir = QPushButton("设置保存目录")
        self.btn_set_exp_dir.clicked.connect(self._on_set_exp_dir)
        dir_layout.addWidget(self.btn_set_exp_dir)
        run_layout.addLayout(dir_layout)

        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("经验文件路径:"))
        self.lbl_exp_path = QLabel("")
        path_layout.addWidget(self.lbl_exp_path)
        path_layout.addStretch()
        run_layout.addLayout(path_layout)

        # ----- 本轮训练统计面板 -----
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
            row, col = idx // cols, idx % cols
            stats_layout.addWidget(QLabel(label_text), row, col * 2)
            stats_layout.addWidget(value_label, row, col * 2 + 1)
        run_layout.addWidget(self.stats_group)

        # ----- 本次训练汇总面板 -----
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
        cols_s = 2
        for idx, (label_text, value_label) in enumerate(summary_items):
            row, col = idx // cols_s, idx % cols_s
            summary_layout.addWidget(QLabel(label_text), row, col * 2)
            summary_layout.addWidget(value_label, row, col * 2 + 1)
        run_layout.addWidget(self.summary_group)

        # ----- 训练控制按钮 -----
        btn_train_layout = QHBoxLayout()
        self.btn_start = QPushButton("开始训练")
        self.btn_start.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
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

        for btn in (self.btn_start, self.btn_pause, self.btn_resume, self.btn_stop,
                    self.btn_save_exp, self.btn_load_exp, self.btn_continuous_train):
            btn_train_layout.addWidget(btn)
        run_layout.addLayout(btn_train_layout)

        # ----- 快速模式切换（无头训练开关）-----
        fast_layout = QHBoxLayout()
        self.cb_fast_mode = QCheckBox("⚡ 快速模式（无渲染）")
        self.cb_fast_mode.setStyleSheet("font-weight: bold; font-size: 13px;")
        fast_layout.addWidget(self.cb_fast_mode)

        fast_layout.addWidget(QLabel("并行环境数:"))
        self.spin_fast_envs = QSpinBox()
        self.spin_fast_envs.setRange(1, 16)
        self.spin_fast_envs.setValue(4)
        fast_layout.addWidget(self.spin_fast_envs)

        fast_layout.addWidget(QLabel("日志间隔(轮):"))
        self.spin_fast_log_interval = QSpinBox()
        self.spin_fast_log_interval.setRange(1, 500)
        self.spin_fast_log_interval.setValue(10)
        fast_layout.addWidget(self.spin_fast_log_interval)

        fast_layout.addStretch()
        run_layout.addLayout(fast_layout)

        layout.addWidget(group_run)
        layout.addStretch()

        self._update_info_labels()

    def _connect_signals(self):
        self.edit_start_size.valueChanged.connect(self._update_start_goal_vis)
        self.edit_goal_size.valueChanged.connect(self._update_start_goal_vis)

    # =========================================================
    # 经验目录初始化 & 显示
    # =========================================================
    def _init_save_dir(self):
        ct = self.config_tab
        if ct.current_config_name and ct.current_mountain_name:
            config_dir = self.main_window.file_manager.get_uav_alg_dir_path(
                ct.current_mountain_name, ct.current_config_name)
            self.save_dir = Path(config_dir).resolve()
        else:
            settings = QSettings("UAVLab", "UAVSimulator")
            last_dir = settings.value("last_exp_dir", "")
            if last_dir and Path(last_dir).exists():
                self.save_dir = Path(last_dir)
            else:
                self.save_dir = Path.home() / "Desktop" / "算法训练集" / "经验保存"
        self._update_exp_dir_display()

    def _update_exp_dir_display(self):
        if self.save_dir:
            self.lbl_exp_dir.setText(str(self.save_dir / "checkpoint"))
            buffer_path = self.save_dir / "checkpoint" / "buffer.pkl"
            self.lbl_exp_path.setText(str(buffer_path))
            if buffer_path.exists():
                self.lbl_exp_path.setToolTip(f"文件大小: {buffer_path.stat().st_size} 字节")
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
            self.train_controller.change_save_dir(
                new_dir, save_current=(reply == QMessageBox.Yes))
        else:
            self.save_dir = new_dir
            QSettings("UAVLab", "UAVSimulator").setValue("last_exp_dir", str(new_dir))
            self._update_exp_dir_display()
            if self.config_tab.current_config_name:
                self.btn_load_exp.setEnabled(True)
                self.btn_continuous_train.setEnabled(True)

    # =========================================================
    # 辅助方法
    # =========================================================
    def _update_info_labels(self):
        ct = self.config_tab
        mountain_text = ct.current_mountain_name or "未加载"
        algo_text = ct.current_config_name or "未加载"
        self.label_current_mountain.setText(mountain_text)
        self.label_current_algorithm.setText(algo_text)
        self.lbl_mountain_info.setText(
            f"已加载: {ct.current_mountain_name}" if ct.current_mountain_name else "未加载山区数据"
        )
        self.lbl_config_info.setText(
            f"已加载: {ct.current_config_name}" if ct.current_config_name else "未加载算法配置"
        )

    def _update_start_goal_vis(self):
        if not hasattr(self.main_window, 'renderer'):
            return
        ct = self.config_tab
        try:
            self.main_window.renderer.draw_start(
                (float(ct.edit_start_x.text()),
                 float(ct.edit_start_y.text()),
                 float(ct.edit_start_z.text())),
                size=self.edit_start_size.value(), color='green'
            )
            self.main_window.renderer.draw_goal(
                (float(ct.edit_goal_x.text()),
                 float(ct.edit_goal_y.text()),
                 float(ct.edit_goal_z.text())),
                size=self.edit_goal_size.value(), color='red'
            )
        except Exception:
            pass

    def _render_mountain_from_config(self):
        """渲染配置页当前山区数据"""
        ct = self.config_tab
        t = ct.current_mountain_data.terrain
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
        self.main_window.renderer.clear_trajectory()
        self.main_window.render_terrain(X, Y, Z)
        for obs in ct.current_mountain_data.obstacles:
            self.main_window.render_obstacle(obs)

    def _get_next_run_index(self, mountain_name, config_name):
        base_dir = self.main_window.file_manager.get_uav_alg_dir_path(mountain_name, config_name)
        if not os.path.exists(base_dir):
            return 1
        max_idx = 0
        pattern = re.compile(r"第(\d+)次训练过程")
        for name in os.listdir(base_dir):
            m = pattern.match(name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        return max_idx + 1

    def _reset_summary_labels(self):
        for lbl in (self.lbl_avg_success, self.lbl_avg_reward, self.lbl_collision_rate,
                    self.lbl_avg_path_length, self.lbl_avg_smoothness, self.lbl_total_epochs,
                    self.lbl_avg_reward_dist, self.lbl_avg_reward_smooth,
                    self.lbl_avg_reward_collision, self.lbl_avg_reward_goal):
            lbl.setText("-")

    # =========================================================
    # 快速加载（独立入口，同步到配置页状态）
    # =========================================================
    def _on_load_mountain(self):
        """在训练页直接加载山区，结果同步到配置页"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择山区文件夹", self.main_window.file_manager.get_root_dir())
        if not dir_path:
            return
        data_path = os.path.join(dir_path, "山区建模数据.json")
        if not os.path.exists(data_path):
            QMessageBox.warning(self, "警告", "所选文件夹中未找到山区建模数据.json")
            return
        try:
            from data.models import MountainData
            data = load_json(data_path)
            mountain_data = MountainData.from_dict(data)
            mountain_name = os.path.basename(dir_path)

            # 同步到配置页
            self.config_tab.current_mountain_data = mountain_data
            self.config_tab.current_mountain_name = mountain_name
            self.config_tab.label_mountain_info.setText(f"已加载: {mountain_name}")

            self._update_info_labels()
            self._render_mountain_from_config()
            self._update_start_goal_vis()
            self.main_window.log(f"山区数据已加载: {mountain_name}")
            QMessageBox.information(self, "成功", f"山区 '{mountain_name}' 加载成功")
        except Exception as e:
            self.main_window.log(f"加载山区数据失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"加载山区数据失败：{str(e)}")

    def _on_load_config(self):
        """在训练页直接加载算法配置，结果同步到配置页"""
        ct = self.config_tab
        if not ct.current_mountain_name:
            QMessageBox.warning(self, "警告", "请先加载一个山区")
            return
        mountain_path = self.main_window.file_manager.get_mountain_path(ct.current_mountain_name)
        if not os.path.exists(mountain_path):
            return
        config_dirs = [d for d in os.listdir(mountain_path)
                       if os.path.isdir(os.path.join(mountain_path, d)) and d.startswith("无人机与算法_")]
        if not config_dirs:
            QMessageBox.information(self, "提示", "该山区下没有找到任何配置文件夹")
            return
        from PyQt5.QtWidgets import QInputDialog
        config_name, ok = QInputDialog.getItem(
            self, "选择配置", "请选择一个配置:", config_dirs, 0, False)
        if not ok or not config_name:
            return
        file_path = self.main_window.file_manager.get_uav_alg_config_path(
            ct.current_mountain_name, config_name)
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "警告", f"配置文件不存在：{file_path}")
            return
        try:
            data = load_json(file_path)
            job_config = TrainJobConfig.from_dict(data)

            # 将参数回填到配置页
            u = job_config.uav
            ct.edit_start_x.setText(str(u.start_x))
            ct.edit_start_y.setText(str(u.start_y))
            ct.edit_start_z.setText(str(u.start_z))
            ct.edit_goal_x.setText(str(u.goal_x))
            ct.edit_goal_y.setText(str(u.goal_y))
            ct.edit_goal_z.setText(str(u.goal_z))
            ct.edit_max_speed.setText(str(u.max_speed))
            ct.edit_max_accel.setText(str(u.max_accel))
            ct.edit_min_alt.setText(str(u.min_altitude))
            ct.edit_max_alt.setText(str(u.max_altitude))
            idx = ct.cb_min_alt_ref.findData(u.min_altitude_ref)
            if idx >= 0:
                ct.cb_min_alt_ref.setCurrentIndex(idx)
            idx = ct.cb_max_alt_ref.findData(u.max_altitude_ref)
            if idx >= 0:
                ct.cb_max_alt_ref.setCurrentIndex(idx)

            a = job_config.algorithm
            ct.edit_lr.setText(str(a.learning_rate))
            ct.edit_gamma.setText(str(a.gamma))
            ct.edit_batch.setValue(a.batch_size)
            ct.edit_hidden_size.setValue(a.hidden_size)
            ct.edit_buffer_capacity.setValue(a.buffer_capacity)
            ct.edit_r_dist.setText(str(a.reward_dist))
            ct.edit_r_collision.setText(str(a.reward_collision))
            ct.edit_r_smooth.setText(str(a.reward_smooth))
            ct.cb_coop.setChecked(a.cooperative)
            ct.cb_use_global.setChecked(getattr(a, 'use_global_path', True))
            ct.spin_step_size.setValue(getattr(a, 'global_step_size', 15.0))
            ct.spin_cylinder_radius.setValue(getattr(a, 'cylinder_radius', 20.0))
            ct.spin_switch_thresh.setValue(getattr(a, 'switch_threshold', 5.0))
            ct.spin_reward_follow.setValue(getattr(a, 'reward_path_follow', -0.5))
            ct.spin_reward_progress.setValue(getattr(a, 'reward_path_progress', 1.0))
            ct.spin_out_penalty.setValue(getattr(a, 'reward_out_of_cylinder', -5.0))

            t = job_config.training
            self.edit_epochs.setValue(t.num_epochs)
            self.edit_save_interval.setValue(t.save_interval)
            self.edit_max_steps.setValue(t.max_steps)

            ct.current_mountain_data = job_config.mountain
            ct.current_mountain_name = os.path.basename(
                os.path.dirname(os.path.dirname(file_path)))
            ct.current_config_name = config_name
            ct.label_mountain_info.setText(f"已加载: {ct.current_mountain_name}")

            self._update_info_labels()
            self._render_mountain_from_config()
            self._update_start_goal_vis()

            # 同步经验目录
            config_dir = self.main_window.file_manager.get_uav_alg_dir_path(
                ct.current_mountain_name, ct.current_config_name)
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

    # =========================================================
    # 经验文件操作
    # =========================================================
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
        target_path = self.save_dir / "checkpoint" / "buffer.pkl"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(file_path, 'rb') as f:
                loaded_buffer = pickle.load(f)
            buffer_capacity = self.config_tab.edit_buffer_capacity.value()
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
            self.main_window.log(
                f"经验文件已加载到: {target_path}, 包含 {len(loaded_buffer)} 条经验")
            QMessageBox.information(
                self, "成功",
                f"已加载经验文件，包含 {len(loaded_buffer)} 条经验。\n经验将保存在：{target_path}")
        except Exception as e:
            self.main_window.log(f"加载经验文件失败: {e}")
            QMessageBox.critical(self, "错误", f"加载失败：{e}")

    def _on_save_experience(self):
        if self.train_controller and self.train_controller.is_training:
            self.train_controller.save_experience_only()
        else:
            QMessageBox.information(self, "提示", "请先开始训练，训练中可保存当前经验")

    # =========================================================
    # 构建完整 JobConfig（合并配置页参数 + 训练页参数）
    # =========================================================
    def _build_full_job_config(self) -> TrainJobConfig:
        job = self.config_tab.get_job_config()
        # 用训练页的训练参数覆盖 TrainConfig 默认值
        from data.models import TrainConfig
        job.training = TrainConfig(
            num_epochs=int(self.edit_epochs.value()),
            save_interval=int(self.edit_save_interval.value()),
            max_steps=self.edit_max_steps.value()
        )
        return job

    # =========================================================
    # 训练控制
    # =========================================================
    def _on_continuous_train(self):
        # 快速模式 → 启动无头脚本（持续模式）
        if self.cb_fast_mode.isChecked():
            self._launch_headless(continuous=True)
            return

        ct = self.config_tab
        if not ct.current_mountain_name:
            reply = QMessageBox.warning(
                self, "未加载山区",
                "当前未加载任何山区数据，将使用默认山区进行训练。\n是否继续？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
            ct.current_mountain_name = "默认山区"
            ct.current_mountain_data = MountainData()
        if not ct.current_config_name:
            ct.current_config_name = "无人机与算法_1"

        self._update_info_labels()
        run_index = self._get_next_run_index(ct.current_mountain_name, ct.current_config_name)
        job_config = self._build_full_job_config()
        self._create_train_controller(job_config, run_index)
        self.epoch_stats_list = []
        self._reset_summary_labels()
        self.train_controller.continuous_mode = True
        self.train_controller.start_training()

    def _launch_headless(self, continuous=False):
        """启动无头快速训练（在新终端窗口运行）
        
        Args:
            continuous: 是否持续训练模式（直到到达终点）
        """
        ct = self.config_tab
        mountain = ct.current_mountain_name
        config_name = ct.current_config_name

        if not mountain:
            mountain = "默认山区"
            ct.current_mountain_name = mountain
            ct.current_mountain_data = MountainData()
        if not config_name:
            config_name = "无人机与算法_1"
            ct.current_config_name = config_name

        # 先将当前配置保存到 JSON（确保脚本读到最新参数）
        from data.serializer import save_json
        job_config = self._build_full_job_config()
        config_path = self.main_window.file_manager.get_uav_alg_config_path(mountain, config_name)
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        save_json(job_config.to_dict(), config_path)

        # 脚本路径
        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script_path = os.path.join(src_dir, "scripts", "train_headless.py")

        if not os.path.exists(script_path):
            QMessageBox.critical(self, "错误", f"快速训练脚本不存在: {script_path}")
            return

        cmd = [
            sys.executable, script_path,
            "--mountain", mountain,
            "--config", config_name,
            "--epochs", str(self.edit_epochs.value()),
            "--log-interval", str(self.spin_fast_log_interval.value()),
            "--save-interval", str(self.edit_save_interval.value()),
            "--envs", str(self.spin_fast_envs.value()),
        ]

        if continuous:
            cmd.append("--epochs")
            cmd.append("99999")  # 持续模式给个大轮数

        try:
            if sys.platform == "win32":
                subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.Popen(cmd)
            mode = "持续" if continuous else "常规"
            self.main_window.log(
                f"[快速{mode}训练已启动] {mountain}/{config_name} "
                f"envs={self.spin_fast_envs.value()} "
                f"epochs={self.edit_epochs.value()}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动快速训练失败: {e}")

    def _on_start_training(self):
        # 快速模式 → 启动无头脚本
        if self.cb_fast_mode.isChecked():
            self._launch_headless(continuous=False)
            return

        ct = self.config_tab
        if not ct.current_mountain_name:
            reply = QMessageBox.warning(
                self, "未加载山区",
                "当前未加载任何山区数据，将使用默认山区进行训练。\n是否继续？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
            ct.current_mountain_name = "默认山区"
            ct.current_mountain_data = MountainData()
        if not ct.current_config_name:
            ct.current_config_name = "无人机与算法_1"

        self._update_info_labels()
        run_index = self._get_next_run_index(ct.current_mountain_name, ct.current_config_name)
        job_config = self._build_full_job_config()
        self._create_train_controller(job_config, run_index)
        self.epoch_stats_list = []
        self._reset_summary_labels()
        self.train_controller.continuous_mode = False
        self.train_controller.start_training()

    def _create_train_controller(self, job_config, run_index):
        ct = self.config_tab
        self.train_controller = TrainController(
            main_window=self.main_window,
            file_manager=self.main_window.file_manager,
            job_config=job_config,
            mountain_name=ct.current_mountain_name,
            config_name=ct.current_config_name,
            run_index=run_index,
            save_dir=self.save_dir
        )
        self.train_controller.training_started.connect(self._on_training_started)
        self.train_controller.training_paused.connect(self._on_training_paused)
        self.train_controller.training_stopped.connect(self._on_training_stopped)
        self.train_controller.epoch_completed.connect(self._on_epoch_completed)
        self.train_controller.trajectory_generated.connect(self._on_trajectory_generated)

    def _on_pause_training(self):
        if self.train_controller:
            self.train_controller.pause_training()

    def _on_resume_training(self):
        if self.train_controller:
            self.train_controller.resume_training()
            self.btn_pause.setEnabled(True)
            self.btn_resume.setEnabled(False)

    def _on_stop_training(self):
        if self.train_controller:
            self.train_controller.stop_training()

    # =========================================================
    # 训练信号回调
    # =========================================================
    def _on_training_started(self):
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_resume.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_save_exp.setEnabled(True)
        self.btn_load_exp.setEnabled(False)
        self.btn_continuous_train.setEnabled(False)
        self.main_window.log("【训练】训练已开始")

    def _on_training_paused(self):
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(True)
        self.main_window.log("【训练】训练已暂停")

    def _on_training_stopped(self):
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_stop.setEnabled(False)
        has_config = bool(self.config_tab.current_config_name)
        self.btn_save_exp.setEnabled(False)
        self.btn_load_exp.setEnabled(has_config)
        self.btn_continuous_train.setEnabled(has_config)
        self.train_controller = None
        self.main_window.log("【训练】训练已停止")

        for lbl in (self.lbl_success, self.lbl_collision, self.lbl_final_dist,
                    self.lbl_path_length, self.lbl_avg_acc, self.lbl_acc_var,
                    self.lbl_total_reward, self.lbl_steps, self.lbl_loss,
                    self.lbl_reward_dist, self.lbl_reward_smooth,
                    self.lbl_reward_collision, self.lbl_reward_goal):
            lbl.setText("-")

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

        self.main_window.log(
            f"【训练】第{epoch}轮: 成功={success}, 碰撞={collision}, "
            f"奖励={total_reward:.2f}, 终止原因={reason}")

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
        s_list = self.epoch_stats_list
        avg_success = sum(1 for s in s_list if s.get('success', False)) / n
        avg_reward = np.mean([s.get('total_reward', 0.0) for s in s_list])
        collision_rate = sum(1 for s in s_list if s.get('collision', False)) / n
        avg_path_length = np.mean([s.get('path_length', 0.0) for s in s_list])
        avg_smoothness = np.mean([s.get('acceleration_variance', 0.0) for s in s_list])
        avg_reward_dist = np.mean([s.get('reward_dist', 0.0) for s in s_list])
        avg_reward_smooth = np.mean([s.get('reward_smooth', 0.0) for s in s_list])
        avg_reward_collision = np.mean([s.get('reward_collision', 0.0) for s in s_list])
        avg_reward_goal = np.mean([s.get('reward_goal', 0.0) for s in s_list])

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

        self.main_window.log(
            f"【训练汇总】共{n}轮，成功率={avg_success:.2%}，"
            f"碰撞率={collision_rate:.2%}，平均奖励={avg_reward:.2f}")
