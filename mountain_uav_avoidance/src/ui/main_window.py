import sys
import numpy as np
import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QGroupBox, QSplitter, QTextEdit,
    QPushButton, QSlider, QFileDialog
)
from PyQt5.QtCore import Qt, QSettings
from pyvista import examples

from ui.tab_mountain import TabMountain
from ui.tab_train import TabTrain
from ui.tab_demo import TabDemo
from visual.renderer_pyvista import PyVistaRenderer
from business.file_manager import FileManager


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("山区无人机避障仿真 (PyVista版)")
        self.setGeometry(100, 100, 1500, 900)

        # 从 QSettings 读取持久化的根目录
        self.settings = QSettings("UAVLab", "MountainUAV")
        saved_root = self.settings.value("train_root", None)
        if saved_root and os.path.exists(saved_root):
            root_dir = saved_root
        else:
            # 默认路径：桌面/算法训练集
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            root_dir = os.path.join(desktop, "算法训练集")

        self.file_manager = FileManager(root_dir)

        self.init_ui()
        # 初始化渲染器后，设置背景并绘制占位地形
        self.renderer.set_background('white')
        self.init_placeholder_terrain()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # 左侧参数配置区
        param_widget = QWidget()
        param_layout = QVBoxLayout(param_widget)
        splitter.addWidget(param_widget)
        splitter.setStretchFactor(0, 1)

        group_root = QGroupBox("全局设置：算法训练集根目录")
        root_layout = QHBoxLayout(group_root)
        self.label_root = QLabel(self.file_manager.get_root_dir())
        self.btn_browse_root = QPushButton("更改目录")
        self.btn_browse_root.clicked.connect(self._on_browse_root)
        root_layout.addWidget(self.label_root)
        root_layout.addWidget(self.btn_browse_root)
        param_layout.addWidget(group_root)

        self.tabs = QTabWidget()
        param_layout.addWidget(self.tabs)

        self.tab1 = TabMountain(self)
        self.tabs.addTab(self.tab1, "山区设置")
        self.tab2 = TabTrain(self)
        self.tabs.addTab(self.tab2, "训练配置")
        self.tab3 = TabDemo(self)
        self.tabs.addTab(self.tab3, "训练演示")

        # 右侧可视化区
        vis_widget = QWidget()
        vis_layout = QVBoxLayout(vis_widget)
        splitter.addWidget(vis_widget)
        splitter.setStretchFactor(1, 2)

        # 创建 PyVista 渲染器并添加到布局
        self.renderer = PyVistaRenderer()
        vis_layout.addWidget(self.renderer)

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(180)
        self.log_text.setReadOnly(True)
        vis_layout.addWidget(self.log_text)

        self.log("PyVista 版启动成功")

    def log(self, msg):
        self.log_text.append(f">> {msg}")

    def _on_browse_root(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择算法训练集根目录",
                                                    self.file_manager.get_root_dir())
        if dir_path:
            self.file_manager.set_root_dir(dir_path)
            self.label_root.setText(dir_path)
            # 保存到 QSettings
            self.settings.setValue("train_root", dir_path)
            self.log(f"训练集根目录已更改为：{dir_path}")

    # ========== 公共渲染接口（保持与原有代码一致） ==========
    def render_terrain(self, X, Y, Z, **kwargs):
        """绘制地形，支持传递额外参数给渲染器"""
        self.renderer.draw_terrain(X, Y, Z, **kwargs)
        self.log("地形已更新至3D画布")

    def render_obstacle(self, obstacle):
        """绘制单个障碍物"""
        self.renderer.draw_obstacle(obstacle)
        self.log("障碍物已添加至3D画布")

    def render_obstacles(self, obstacles):
        """批量绘制障碍物"""
        self.renderer.draw_obstacles(obstacles)
        self.log(f"已绘制 {len(obstacles)} 个障碍物")

    def clear_visualization(self):
        """清空所有绘制内容"""
        self.renderer.clear_all()
        self.log("3D画布已清空")

    def init_placeholder_terrain(self):
        """初始化一个简单的占位地形（类似于之前的示例地形）"""
        x = np.arange(-50, 50, 2)
        y = np.arange(-50, 50, 2)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.sqrt(X**2 + Y**2) / 10) * 5 + 10
        self.render_terrain(X, Y, Z)
        self.log("占位地形已加载")