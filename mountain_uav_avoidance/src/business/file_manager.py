"""
文件目录管理模块

提供基于预设文件架构的路径生成和目录创建功能。
"""

import os


class FileManager:
    """文件目录管理器"""

    def __init__(self, root_dir=None):
        """
        初始化文件管理器

        Args:
            root_dir: 训练集根目录路径。若为None，则使用默认路径（桌面/算法训练集）
        """
        if root_dir is None:
            # 默认路径：桌面/算法训练集
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            self.root_dir = os.path.join(desktop, "算法训练集")
        else:
            self.root_dir = root_dir
        print(f"[FileManager] 初始化，根目录: {self.root_dir}")

    def set_root_dir(self, root_dir):
        """设置新的根目录"""
        self.root_dir = root_dir
        print(f"[FileManager] 根目录已更新为: {self.root_dir}")

    def get_root_dir(self):
        """获取当前根目录"""
        return self.root_dir

    def get_mountain_path(self, mountain_name):
        """获取山区文件夹路径"""
        path = os.path.join(self.root_dir, mountain_name)
        return path

    def get_mountain_data_path(self, mountain_name):
        """获取山区建模数据文件路径"""
        path = os.path.join(self.get_mountain_path(mountain_name), "山区建模数据.json")
        return path

    def get_uav_alg_dir_path(self, mountain_name, config_name):
        """获取无人机与算法配置文件夹路径"""
        path = os.path.join(self.get_mountain_path(mountain_name), config_name)
        return path

    def get_uav_alg_config_path(self, mountain_name, config_name):
        """获取无人机算法数据文件路径"""
        path = os.path.join(self.get_uav_alg_dir_path(mountain_name, config_name), "无人机算法数据.json")
        return path

    def get_train_run_dir_path(self, mountain_name, config_name, run_index):
        """获取某次训练过程文件夹路径"""
        run_dir_name = f"第{run_index}次训练过程"
        path = os.path.join(self.get_uav_alg_dir_path(mountain_name, config_name), run_dir_name)
        return path

    def get_trajectory_path(self, mountain_name, config_name, run_index, episode):
        """获取某轮训练轨迹文件路径"""
        file_name = f"第{episode}轮训练过程.npy"
        path = os.path.join(self.get_train_run_dir_path(mountain_name, config_name, run_index), file_name)
        return path

    def create_train_dirs(self, mountain_name, config_name, run_index):
        """
        创建训练所需的所有目录

        Returns:
            str: 返回训练过程文件夹路径
        """
        path = self.get_train_run_dir_path(mountain_name, config_name, run_index)
        os.makedirs(path, exist_ok=True)
        print(f"[FileManager] 已创建目录: {path}")
        return path


if __name__ == "__main__":
    fm = FileManager()
    print("根目录:", fm.get_root_dir())
    print("山区路径:", fm.get_mountain_path("山区1"))
    print("山区数据路径:", fm.get_mountain_data_path("山区1"))
    print("配置文件夹路径:", fm.get_uav_alg_dir_path("山区1", "无人机与算法_1"))
    print("配置文件路径:", fm.get_uav_alg_config_path("山区1", "无人机与算法_1"))
    print("训练过程文件夹:", fm.get_train_run_dir_path("山区1", "无人机与算法_1", 1))
    print("轨迹文件路径:", fm.get_trajectory_path("山区1", "无人机与算法_1", 1, 10))
    fm.create_train_dirs("山区1", "无人机与算法_1", 1)