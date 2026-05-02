"""
演示流程控制模块

封装演示数据的加载、播放、暂停、进度控制等逻辑。
使用 QTimer 驱动逐帧播放，通过信号与 UI 交互。
修正：滑块拖动时自动暂停，播放速度边界检查。
新增：保存文件路径，data_loaded 信号携带路径。
增强：支持加载扩展轨迹（13维），提供 get_current_info 方法解析详细信息。
"""

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QTimer

from data.serializer import load_npy


class DemoController(QObject):
    """
    演示控制器

    负责加载轨迹数据，控制播放进度，并发射信号更新 UI。
    播放速度可调（默认100ms/帧）。
    支持旧格式（仅位置）和新格式（13维：位置、速度、加速度、离地高度、碰撞、距终点距离、累计路径长度）。
    """

    frame_changed = pyqtSignal(int)          # 当前帧索引
    total_frames_changed = pyqtSignal(int)   # 总帧数
    playback_started = pyqtSignal()
    playback_paused = pyqtSignal()
    playback_stopped = pyqtSignal()
    data_loaded = pyqtSignal(np.ndarray, str)  # 新数据加载完成，携带轨迹和文件路径

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.timer = QTimer()
        self.timer.setInterval(100)  # 默认100ms
        self.timer.timeout.connect(self._on_timer)

        self.trajectory = None
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False
        self.file_path = None  # 保存加载的文件路径
        self.has_full_info = False  # 是否为扩展格式（13维）

    def load_data(self, file_path):
        """加载 .npy 轨迹文件，自动检测格式"""
        try:
            data = load_npy(file_path)
            if data.size == 0:
                self.main_window.log("[DemoController] 文件为空")
                return False

            # 检测轨迹格式
            if data.ndim == 2:
                if data.shape[1] == 3:
                    self.has_full_info = False
                    self.main_window.log("[DemoController] 加载旧格式轨迹（仅位置）")
                elif data.shape[1] >= 13:
                    self.has_full_info = True
                    self.main_window.log("[DemoController] 加载扩展格式轨迹（13维）")
                else:
                    self.main_window.log(f"[DemoController] 未知轨迹维度: {data.shape[1]}，按旧格式处理")
                    self.has_full_info = False
            else:
                self.main_window.log("[DemoController] 轨迹数据维度异常，加载失败")
                return False

            self.trajectory = data
            self.total_frames = len(self.trajectory)
            self.current_frame = 0
            self.file_path = file_path
            self.total_frames_changed.emit(self.total_frames)
            self.frame_changed.emit(self.current_frame)
            self.data_loaded.emit(self.trajectory, file_path)
            self.playback_stopped.emit()  # 确保按钮状态为停止
            self.main_window.log(f"[DemoController] 已加载轨迹，共 {self.total_frames} 帧")
            return True
        except Exception as e:
            self.main_window.log(f"[DemoController] 加载失败: {str(e)}")
            return False

    def play(self):
        if self.trajectory is None or self.total_frames == 0:
            self.main_window.log("[DemoController] 无数据可播放")
            return
        if not self.is_playing:
            self.is_playing = True
            self.timer.start()
            self.playback_started.emit()
            self.main_window.log("[DemoController] 播放")

    def pause(self):
        if self.is_playing:
            self.is_playing = False
            self.timer.stop()
            self.playback_paused.emit()
            self.main_window.log("[DemoController] 暂停")

    def stop(self):
        self.pause()
        self.current_frame = 0
        self.frame_changed.emit(self.current_frame)
        self.playback_stopped.emit()
        self.main_window.log("[DemoController] 停止")

    def set_frame(self, frame_index):
        """跳转到指定帧，若正在播放则自动暂停"""
        if self.trajectory is None:
            return
        # 边界裁剪
        frame_index = max(0, min(frame_index, self.total_frames - 1))
        # 如果正在播放，自动暂停
        if self.is_playing:
            self.pause()
        # 仅当帧改变时才更新
        if frame_index != self.current_frame:
            self.current_frame = frame_index
            self.frame_changed.emit(self.current_frame)

    def set_speed(self, interval_ms):
        """设置播放速度（毫秒/帧），限制在合理范围"""
        interval_ms = max(30, min(interval_ms, 1000))  # 30ms ~ 1000ms
        self.timer.setInterval(interval_ms)

    def _on_timer(self):
        if self.trajectory is None:
            self.pause()
            return
        next_frame = self.current_frame + 1
        if next_frame >= self.total_frames:
            self.stop()
        else:
            self.current_frame = next_frame
            self.frame_changed.emit(self.current_frame)

    def get_current_position(self):
        """获取当前帧的位置（兼容旧格式）"""
        if self.trajectory is not None and 0 <= self.current_frame < self.total_frames:
            row = self.trajectory[self.current_frame]
            if self.has_full_info:
                return row[0:3]  # 前三个是位置
            else:
                return row  # 旧格式就是位置
        return None

    def get_current_info(self):
        """
        获取当前帧的详细信息，返回字典。
        对于旧格式，仅包含位置信息；对于新格式，包含所有字段。
        """
        if self.trajectory is None or self.current_frame < 0 or self.current_frame >= self.total_frames:
            return None

        row = self.trajectory[self.current_frame]

        if self.has_full_info:
            # 新格式：确保至少有13列
            if len(row) >= 13:
                return {
                    'pos': row[0:3],
                    'vel': row[3:6],
                    'acc': row[6:9],
                    'ground_clearance': row[9],
                    'collision': bool(row[10]),
                    'dist_to_goal': row[11],
                    'cumulative_dist': row[12]
                }
            else:
                # 理论上不会发生，但以防万一
                self.main_window.log("[DemoController] 扩展格式但列数不足，降级处理")
                return {
                    'pos': row[0:3] if len(row) >= 3 else np.array([0,0,0]),
                    'vel': None,
                    'acc': None,
                    'ground_clearance': None,
                    'collision': None,
                    'dist_to_goal': None,
                    'cumulative_dist': None
                }
        else:
            # 旧格式：只有位置
            return {
                'pos': row if len(row) == 3 else np.array([0,0,0]),
                'vel': None,
                'acc': None,
                'ground_clearance': None,
                'collision': None,
                'dist_to_goal': None,
                'cumulative_dist': None
            }

    def has_data(self):
        return self.trajectory is not None and self.total_frames > 0