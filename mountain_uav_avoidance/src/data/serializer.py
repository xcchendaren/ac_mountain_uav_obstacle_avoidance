"""
数据序列化工具模块

包含 JSON 和 NumPy 数组的实际保存/加载方法。
"""

import json
import numpy as np
import os


def save_json(data, file_path):
    """
    将 Python 对象保存为 JSON 文件

    Args:
        data: 可序列化为 JSON 的 Python 对象（通常为 dict）
        file_path: 保存路径
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[serializer] JSON saved to {file_path}")


def load_json(file_path):
    """
    从 JSON 文件加载 Python 对象

    Args:
        file_path: JSON 文件路径
    Returns:
        dict: 加载的数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_npy(data, file_path):
    """
    将 NumPy 数组保存为 .npy 文件

    Args:
        data: numpy.ndarray 对象
        file_path: 保存路径
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, data)
    print(f"[serializer] NPY saved to {file_path}")


def load_npy(file_path):
    """
    从 .npy 文件加载 NumPy 数组

    Args:
        file_path: .npy 文件路径
    Returns:
        numpy.ndarray: 加载的数组
    """
    return np.load(file_path)


if __name__ == "__main__":
    # 简单的自我测试
    test_data = {"key": "value"}
    save_json(test_data, "test.json")
    loaded = load_json("test.json")
    print("加载结果:", loaded)

    arr = np.array([1, 2, 3])
    save_npy(arr, "test.npy")
    loaded_arr = load_npy("test.npy")
    print("加载数组:", loaded_arr)