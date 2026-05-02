import sys
import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QGroupBox, QSplitter, QTextEdit,
    QPushButton, QComboBox, QSlider, QCheckBox, QLineEdit, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class FullFileDialogDemo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("山区无人机避障仿真 - 文件对话框交互版")
        self.setGeometry(100, 100, 1500, 900)

        # 初始化默认路径
        self.default_train_root = os.path.join(os.path.expanduser("~"), "Desktop", "算法训练集")
        self.current_train_root = self.default_train_root

        self.init_ui()
        self.init_3d_view()

    def init_ui(self):
        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # --- 左侧：参数配置区 ---
        param_widget = QWidget()
        param_layout = QVBoxLayout(param_widget)
        splitter.addWidget(param_widget)
        splitter.setStretchFactor(0, 1)

        # 【新增】全局：训练集根目录设置
        group_root = QGroupBox("全局设置：算法训练集根目录")
        root_layout = QHBoxLayout(group_root)
        self.label_root = QLabel(self.current_train_root)
        self.label_root.setStyleSheet("color: blue;")
        self.btn_browse_root = QPushButton("更改目录")
        self.btn_browse_root.clicked.connect(self.browse_train_root)
        root_layout.addWidget(self.label_root)
        root_layout.addWidget(self.btn_browse_root)
        param_layout.addWidget(group_root)

        # 标签页容器
        self.tabs = QTabWidget()
        param_layout.addWidget(self.tabs)

        # 1. 标签页 1：山区设置
        self.tab1 = QWidget()
        self.tabs.addTab(self.tab1, "山区设置")
        self.create_tab1()

        # 2. 标签页 2：训练配置
        self.tab2 = QWidget()
        self.tabs.addTab(self.tab2, "训练配置")
        self.create_tab2()

        # 3. 标签页 3：训练演示
        self.tab3 = QWidget()
        self.tabs.addTab(self.tab3, "训练演示")
        self.create_tab3()

        # --- 右侧：可视化区 ---
        vis_widget = QWidget()
        vis_layout = QVBoxLayout(vis_widget)
        splitter.addWidget(vis_widget)
        splitter.setStretchFactor(1, 2)

        # Matplotlib 画布
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        vis_layout.addWidget(self.canvas)

        # 日志窗口
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(180)
        self.log_text.setReadOnly(True)
        vis_layout.addWidget(self.log_text)

        self.log("软件启动成功！")
        self.log(f"默认训练集路径：{self.current_train_root}")

    def log(self, msg):
        self.log_text.append(f">> {msg}")

    # ==================== 全局路径设置 ====================
    def browse_train_root(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择算法训练集根目录", os.path.expanduser("~"))
        if dir_path:
            self.current_train_root = dir_path
            self.label_root.setText(dir_path)
            self.log(f"训练集根目录已更改为：{dir_path}")

    # ==================== 标签页 1：山区设置 ====================
    def create_tab1(self):
        layout = QVBoxLayout(self.tab1)

        # 上：地形设置
        group1 = QGroupBox("山区地形设置")
        g1_layout = QVBoxLayout(group1)
        g1_layout.addWidget(QLabel("(此处放置：公式系数、地形大小输入控件)"))
        layout.addWidget(group1)

        # 下：障碍物设置
        group2 = QGroupBox("障碍物设置")
        g2_layout = QVBoxLayout(group2)
        g2_layout.addWidget(QLabel("(此处放置：障碍类型、位置、大小、列表控件)"))
        layout.addWidget(group2)

        # 操作按钮
        btn_layout = QHBoxLayout()
        btn1 = QPushButton("保存山区")
        btn2 = QPushButton("加载山区")
        btn3 = QPushButton("删除山区")
        btn1.clicked.connect(self.save_mountain)
        btn2.clicked.connect(self.load_mountain)
        btn3.clicked.connect(self.delete_mountain)
        btn_layout.addWidget(btn1)
        btn_layout.addWidget(btn2)
        btn_layout.addWidget(btn3)
        layout.addLayout(btn_layout)
        layout.addStretch()

    def save_mountain(self):
        # 1. 确保根目录存在
        if not os.path.exists(self.current_train_root):
            os.makedirs(self.current_train_root)

        # 2. 选择保存的子文件夹（或输入名称）
        dir_path = QFileDialog.getExistingDirectory(self, "选择保存位置（或在根目录下新建文件夹）",
                                                    self.current_train_root)
        if dir_path:
            # 模拟保存
            save_path = os.path.join(dir_path, "山区建模数据.json")
            self.log(f"正在保存山区数据至：{save_path}")
            QMessageBox.information(self, "成功", f"山区数据已保存至：\n{save_path}")

    def load_mountain(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择要加载的山区文件夹", self.current_train_root)
        if dir_path:
            self.log(f"正在从以下位置加载山区数据：{dir_path}")
            QMessageBox.information(self, "成功", f"已加载山区数据：\n{os.path.basename(dir_path)}")

    def delete_mountain(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择要删除的山区文件夹", self.current_train_root)
        if dir_path:
            reply = QMessageBox.question(self, "确认删除", f"确定要删除以下文件夹吗？\n{dir_path}",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.log(f"已删除山区文件夹：{dir_path}")
                QMessageBox.information(self, "已删除", "文件夹已删除（演示模式，未真实删除）")

    # ==================== 标签页 2：训练配置 ====================
    def create_tab2(self):
        layout = QVBoxLayout(self.tab2)

        # 1. 加载与选择
        group_load = QGroupBox("加载山区模型")
        load_layout = QHBoxLayout(group_load)
        self.btn_load_mountain_dir = QPushButton("选择并加载山区文件夹")
        self.btn_load_mountain_dir.clicked.connect(self.load_mountain_dir_for_train)
        load_layout.addWidget(self.btn_load_mountain_dir)
        layout.addWidget(group_load)

        # 2. 无人机与算法设置
        group_uav = QGroupBox("无人机设置")
        uav_layout = QVBoxLayout(group_uav)
        uav_layout.addWidget(QLabel("(此处放置：起点/终点、物理约束)"))
        layout.addWidget(group_uav)

        group_alg = QGroupBox("算法设置")
        alg_layout = QVBoxLayout(group_alg)
        alg_layout.addWidget(QLabel("(此处放置：超参数、奖励函数、协同机制复选框)"))
        layout.addWidget(group_alg)

        # 3. 配置操作按钮
        btn_config_layout = QHBoxLayout()
        btn_save_cfg = QPushButton("保存配置")
        btn_load_cfg = QPushButton("加载配置")
        btn_del_cfg = QPushButton("删除配置")
        btn_save_cfg.clicked.connect(self.save_uav_alg_config)
        btn_load_cfg.clicked.connect(self.load_uav_alg_config)
        btn_del_cfg.clicked.connect(self.delete_uav_alg_config)
        btn_config_layout.addWidget(btn_save_cfg)
        btn_config_layout.addWidget(btn_load_cfg)
        btn_config_layout.addWidget(btn_del_cfg)
        layout.addLayout(btn_config_layout)

        # 4. 训练控制
        group_train = QGroupBox("训练设置")
        train_layout = QVBoxLayout(group_train)
        train_layout.addWidget(QLabel("(此处放置：训练轮数、保存间隔等)"))
        layout.addWidget(group_train)

        btn_start = QPushButton("开始训练")
        btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        btn_start.clicked.connect(self.start_training)
        layout.addWidget(btn_start)
        layout.addStretch()

    def load_mountain_dir_for_train(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择山区文件夹", self.current_train_root)
        if dir_path:
            self.log(f"已选择山区用于训练：{dir_path}")

    def save_uav_alg_config(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "保存无人机&算法配置", self.current_train_root,
                                                   "JSON Files (*.json)")
        if file_path:
            self.log(f"配置已保存至：{file_path}")
            QMessageBox.information(self, "成功", "配置文件已保存！")

    def load_uav_alg_config(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "加载无人机&算法配置", self.current_train_root,
                                                   "JSON Files (*.json)")
        if file_path:
            self.log(f"已加载配置文件：{file_path}")

    def delete_uav_alg_config(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择要删除的配置文件", self.current_train_root,
                                                   "JSON Files (*.json)")
        if file_path:
            reply = QMessageBox.question(self, "确认", f"删除此文件？\n{file_path}", QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.log(f"已删除配置：{file_path}")

    def start_training(self):
        self.log("=" * 40)
        self.log(">> 开始训练！")
        self.log(f">> 数据将保存至：{self.current_train_root}/.../第X次训练过程/")
        self.log("=" * 40)
        QMessageBox.information(self, "训练中", "训练已开始（演示模式）")

    # ==================== 标签页 3：训练演示 ====================
    def create_tab3(self):
        layout = QVBoxLayout(self.tab3)

        group_demo = QGroupBox("演示控制")
        demo_layout = QVBoxLayout(group_demo)

        btn_row = QHBoxLayout()
        btn_load_demo = QPushButton("加载演示数据（选择 .npy 文件）")
        btn_play = QPushButton("播放")
        btn_pause = QPushButton("暂停")
        btn_load_demo.clicked.connect(self.load_demo_file)
        btn_row.addWidget(btn_load_demo)
        btn_row.addWidget(btn_play)
        btn_row.addWidget(btn_pause)
        demo_layout.addLayout(btn_row)

        demo_layout.addWidget(QLabel("训练轮次/进度："))
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(1)
        slider.setMaximum(100)
        demo_layout.addWidget(slider)

        layout.addWidget(group_demo)
        layout.addStretch()

    def load_demo_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择训练过程文件", self.current_train_root,
                                                   "Numpy Files (*.npy);;All Files (*)")
        if file_path:
            self.log(f"已加载演示数据：{file_path}")

    # ==================== 3D 视图 ====================
    def init_3d_view(self):
        self.ax.clear()
        self.ax.set_title("3D 可视化区域")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        x = np.arange(-50, 50, 2)
        y = np.arange(-50, 50, 2)
        x, y = np.meshgrid(x, y)
        z = np.sin(np.sqrt(x ** 2 + y ** 2) / 10) * 5 + 10
        self.ax.plot_surface(x, y, z, cmap='terrain', alpha=0.5)
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FullFileDialogDemo()
    window.show()
    sys.exit(app.exec_())