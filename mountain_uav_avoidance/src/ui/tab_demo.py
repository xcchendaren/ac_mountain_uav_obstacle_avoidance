from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QSlider, QFileDialog, QMessageBox, QFormLayout
)
from PyQt5.QtCore import Qt
import os
import numpy as np

from business.demo_controller import DemoController
from business.uav_builder import UAV
from data.models import UAVConfig, MountainData
from data.serializer import load_json
from business.terrain_builder import generate_terrain, generate_terrain_perlin


class TabDemo(QWidget):
    """训练演示标签页：绑定演示控制器，实现数据加载、播放、进度控制，并显示增强的轨迹信息"""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.demo_controller = DemoController(main_window)
        self.has_full_info = False  # 当前加载的轨迹是否包含完整信息（13列）
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        layout = QVBoxLayout(self)

        group_demo = QGroupBox("演示控制")
        demo_layout = QVBoxLayout(group_demo)

        # ===== 第一行：加载按钮 + 文件路径显示 =====
        top_row = QHBoxLayout()
        self.btn_load_demo = QPushButton("加载演示数据")
        self.file_path_label = QLabel("未加载")          # 显示当前文件路径
        self.file_path_label.setWordWrap(True)
        self.file_path_label.setMinimumWidth(300)
        top_row.addWidget(self.btn_load_demo)
        top_row.addWidget(self.file_path_label)
        top_row.addStretch()
        demo_layout.addLayout(top_row)

        # ===== 第二行：播放控制按钮 =====
        btn_row = QHBoxLayout()
        self.btn_play = QPushButton("播放")
        self.btn_pause = QPushButton("暂停")
        self.btn_stop = QPushButton("停止")
        self.btn_speed_slow = QPushButton("慢速")
        self.btn_speed_normal = QPushButton("中速")
        self.btn_speed_fast = QPushButton("快速")

        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)

        btn_row.addWidget(self.btn_play)
        btn_row.addWidget(self.btn_pause)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_speed_slow)
        btn_row.addWidget(self.btn_speed_normal)
        btn_row.addWidget(self.btn_speed_fast)
        demo_layout.addLayout(btn_row)

        # ===== 进度条 =====
        demo_layout.addWidget(QLabel("演示进度："))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        demo_layout.addWidget(self.slider)

        # ===== 增强信息显示面板 =====
        self.info_group = QGroupBox("实时轨迹信息")
        info_layout = QFormLayout(self.info_group)
        self.lbl_frame = QLabel("0/0")
        self.lbl_pos = QLabel("(0.0, 0.0, 0.0)")
        self.lbl_vel = QLabel("(0.0, 0.0, 0.0)")
        self.lbl_acc = QLabel("(0.0, 0.0, 0.0)")
        self.lbl_ground = QLabel("0.0")
        self.lbl_dist = QLabel("0.0")
        self.lbl_cumulative = QLabel("0.0")
        self.lbl_collision = QLabel("否")
        info_layout.addRow("帧:", self.lbl_frame)
        info_layout.addRow("位置 (x,y,z):", self.lbl_pos)
        info_layout.addRow("速度 (vx,vy,vz):", self.lbl_vel)
        info_layout.addRow("加速度 (ax,ay,az):", self.lbl_acc)
        info_layout.addRow("离地高度 (m):", self.lbl_ground)
        info_layout.addRow("距终点 (m):", self.lbl_dist)
        info_layout.addRow("已飞距离 (m):", self.lbl_cumulative)
        info_layout.addRow("碰撞:", self.lbl_collision)
        demo_layout.addWidget(self.info_group)

        # 说明
        label_info = QLabel("说明：加载训练过程中保存的 .npy 轨迹文件进行回放")
        label_info.setStyleSheet("color: gray;")
        demo_layout.addWidget(label_info)

        layout.addWidget(group_demo)
        layout.addStretch()

        # 信号连接
        self.btn_load_demo.clicked.connect(self.on_load_demo)
        self.btn_play.clicked.connect(self.on_play)
        self.btn_pause.clicked.connect(self.on_pause)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_speed_slow.clicked.connect(lambda: self.demo_controller.set_speed(300))
        self.btn_speed_normal.clicked.connect(lambda: self.demo_controller.set_speed(100))
        self.btn_speed_fast.clicked.connect(lambda: self.demo_controller.set_speed(30))
        self.slider.sliderMoved.connect(self.on_slider_moved)

    def connect_signals(self):
        self.demo_controller.frame_changed.connect(self.on_frame_changed)
        self.demo_controller.total_frames_changed.connect(self.on_total_frames_changed)
        self.demo_controller.playback_started.connect(self.on_playback_started)
        self.demo_controller.playback_paused.connect(self.on_playback_paused)
        self.demo_controller.playback_stopped.connect(self.on_playback_stopped)
        self.demo_controller.data_loaded.connect(self.on_data_loaded)

    # ---------- 辅助方法：解析文件路径 ----------
    def _parse_path_info(self, file_path):
        """从文件路径解析山区、配置、训练次数、轮次"""
        parts = file_path.replace('\\', '/').split('/')
        filename = parts[-1]
        # 提取轮次
        episode = None
        if filename.startswith('第') and '轮训练过程' in filename:
            ep_str = filename[1:filename.index('轮')]
            if ep_str.isdigit():
                episode = int(ep_str)
        # 训练次数
        train_dir = parts[-2] if len(parts) >= 2 else ''
        run_index = None
        if train_dir.startswith('第') and '次训练过程' in train_dir:
            run_str = train_dir[1:train_dir.index('次')]
            if run_str.isdigit():
                run_index = int(run_str)
        # 配置名称
        config_name = parts[-3] if len(parts) >= 3 else ''
        # 山区名称
        mountain_name = parts[-4] if len(parts) >= 4 else ''
        return mountain_name, config_name, run_index, episode

    # ---------- 加载山区模型 ----------
    def _load_mountain_model(self, mountain_name):
        """根据山区名称加载对应的地形和障碍物（不重置相机）"""
        if not mountain_name:
            self.main_window.log("警告：无法从路径中解析山区名称，保持现有地形")
            return False

        file_path = self.main_window.file_manager.get_mountain_data_path(mountain_name)
        if not os.path.exists(file_path):
            self.main_window.log(f"警告：山区数据文件不存在 {file_path}")
            return False

        try:
            data = load_json(file_path)
            mountain_data = MountainData.from_dict(data)

            # 清除现有地形和障碍物（但保留轨迹线，稍后会重新绘制）
            self.main_window.renderer.clear_terrain()
            self.main_window.renderer.clear_obstacles()

            # 生成并渲染地形（不重置相机）
            t = mountain_data.terrain
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

            # 传递 reset_camera=False 保持当前视角
            self.main_window.render_terrain(X, Y, Z, reset_camera=False)

            # 渲染障碍物
            for obs in mountain_data.obstacles:
                self.main_window.render_obstacle(obs)

            self.main_window.log(f"已加载山区模型: {mountain_name}")
            return True
        except Exception as e:
            self.main_window.log(f"加载山区模型失败: {str(e)}")
            QMessageBox.warning(self, "警告", f"加载山区模型失败：{str(e)}")
            return False

    # ---------- 事件处理 ----------
    def on_load_demo(self):
        """打开文件对话框选择轨迹文件，并加载"""
        start_dir = self.main_window.file_manager.get_root_dir()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择轨迹文件", start_dir,
            "Numpy Files (*.npy);;All Files (*)"
        )
        if file_path:
            success = self.demo_controller.load_data(file_path)
            if success:
                self.main_window.log(f"已加载演示文件: {file_path}")

    def on_play(self):
        self.demo_controller.play()

    def on_pause(self):
        self.demo_controller.pause()

    def on_stop(self):
        self.demo_controller.stop()

    def on_slider_moved(self, value):
        self.demo_controller.set_frame(value)

    def on_frame_changed(self, frame):
        """更新滑块和信息显示"""
        self.slider.setValue(frame)

        # 获取当前帧的详细信息
        info = self.demo_controller.get_current_info()
        if info is None:
            return

        # 更新帧信息
        total = self.demo_controller.total_frames
        self.lbl_frame.setText(f"{frame+1}/{total}")

        # 更新位置（始终存在）
        pos = info.get('pos')
        if pos is not None:
            self.lbl_pos.setText(f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
            # 更新无人机位置
            if hasattr(self.main_window, 'renderer'):
                fake_uav = UAV(UAVConfig())
                fake_uav.position = pos
                self.main_window.renderer.draw_uav(fake_uav, color='red', size=2.0)

        # 更新其他信息（如果存在）
        if info.get('vel') is not None:
            vel = info['vel']
            self.lbl_vel.setText(f"({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f})")
        else:
            self.lbl_vel.setText("N/A")

        if info.get('acc') is not None:
            acc = info['acc']
            self.lbl_acc.setText(f"({acc[0]:.2f}, {acc[1]:.2f}, {acc[2]:.2f})")
        else:
            self.lbl_acc.setText("N/A")

        if info.get('ground_clearance') is not None:
            self.lbl_ground.setText(f"{info['ground_clearance']:.2f}")
        else:
            self.lbl_ground.setText("N/A")

        if info.get('dist_to_goal') is not None:
            self.lbl_dist.setText(f"{info['dist_to_goal']:.2f}")
        else:
            self.lbl_dist.setText("N/A")

        if info.get('cumulative_dist') is not None:
            self.lbl_cumulative.setText(f"{info['cumulative_dist']:.2f}")
        else:
            self.lbl_cumulative.setText("N/A")

        if info.get('collision') is not None:
            self.lbl_collision.setText("是" if info['collision'] else "否")
        else:
            self.lbl_collision.setText("N/A")

    def on_total_frames_changed(self, total):
        self.slider.setMaximum(total - 1 if total > 0 else 0)

    def on_playback_started(self):
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_load_demo.setEnabled(False)

    def on_playback_paused(self):
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_load_demo.setEnabled(False)

    def on_playback_stopped(self):
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_load_demo.setEnabled(True)
        self.main_window.renderer.clear_uav()
        # 清除信息显示（可选）
        self.lbl_frame.setText("0/0")
        self.lbl_pos.setText("(0.0, 0.0, 0.0)")
        self.lbl_vel.setText("N/A")
        self.lbl_acc.setText("N/A")
        self.lbl_ground.setText("N/A")
        self.lbl_dist.setText("N/A")
        self.lbl_cumulative.setText("N/A")
        self.lbl_collision.setText("N/A")

    def on_data_loaded(self, trajectory, file_path):
        """轨迹加载完成后的处理"""
        # 解析路径信息
        mountain, config, run, episode = self._parse_path_info(file_path)

        # 加载对应的山区模型（不重置相机）
        self._load_mountain_model(mountain)

        # 判断轨迹格式：检查列数
        if trajectory.ndim == 2 and trajectory.shape[1] >= 13:
            self.has_full_info = True
            # 绘制轨迹线时只使用位置列（前3列）
            pos_trajectory = trajectory[:, :3]
        else:
            self.has_full_info = False
            pos_trajectory = trajectory if trajectory.ndim == 2 else trajectory.reshape(-1, 3)

        # 绘制完整轨迹（清除已有轨迹后重新绘制）
        self.main_window.renderer.clear_trajectory()
        self.main_window.renderer.draw_trajectory(pos_trajectory, color='green', line_width=2)

        # 更新文件路径显示
        self.file_path_label.setText(file_path)

        # 构建日志信息
        info_text = f"山区: {mountain} | 配置: {config} | 第{run}次训练 | 第{episode}轮 | 共{len(pos_trajectory)}个点"
        if self.has_full_info:
            info_text += " (完整信息)"
        else:
            info_text += " (仅位置)"
        self.main_window.log(f"轨迹已加载: {info_text}")

        # 强制设置为停止状态
        self.on_playback_stopped()