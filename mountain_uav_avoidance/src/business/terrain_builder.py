"""
山区地形建模模块

基于函数建模法或柏林噪声生成三维地形坐标。
新增：柏林噪声生成器，用于更自然的地形模拟。
"""

import numpy as np
from data.models import TerrainData
from noise import pnoise2  # 导入柏林噪声2D函数


def generate_terrain_perlin(terrain_data: TerrainData,
                            scale: float = 50.0,
                            octaves: int = 6,
                            persistence: float = 0.5,
                            lacunarity: float = 2.0,
                            seed: int = None,
                            height_scale: float = 20.0):
    """
    使用柏林噪声生成山区地形。

    Args:
        terrain_data: TerrainData 对象，用于获取网格范围和分辨率。
        scale: 噪声的缩放比例。值越大，地形起伏的“波长”越长，地形越平缓。
        octaves: 倍频程数。叠加的噪声层数，越多细节越丰富，但计算量越大。
        persistence: 持续度。控制每个倍频程的振幅衰减，值越大地形越崎岖。
        lacunarity: 空隙度。控制每个倍频程的频率增加，通常为2.0。
        seed: 随机种子。相同的种子和参数可以生成完全相同的地形。
        height_scale: 高度缩放因子，用于将噪声值（约-1~1）放大到实际地形高度。

    Returns:
        tuple: (X, Y, Z)
    """
    # 获取地形范围和分辨率
    x_min, x_max = terrain_data.x_min, terrain_data.x_max
    y_min, y_max = terrain_data.y_min, terrain_data.y_max
    res = terrain_data.resolution

    # 使用 linspace 生成精确的坐标轴
    num_x = int((x_max - x_min) / res) + 1
    num_y = int((y_max - y_min) / res) + 1
    x = np.linspace(x_min, x_max, num_x)
    y = np.linspace(y_min, y_max, num_y)

    # 生成二维网格
    X, Y = np.meshgrid(x, y)

    # 初始化高度图数组
    Z = np.zeros_like(X)

    # 设置随机种子（如果提供）
    base_seed = seed if seed is not None else np.random.randint(0, 10000)

    # 遍历每个网格点，计算柏林噪声值
    for i in range(num_x):
        for j in range(num_y):
            # 核心：调用 pnoise2 函数
            # 将坐标除以 scale 来调整噪声的“密度”
            # 使用不同的种子值作为z参数来生成不同的2D切片
            noise_val = pnoise2(x[i] / scale,
                                y[j] / scale,
                                octaves=octaves,
                                persistence=persistence,
                                lacunarity=lacunarity,
                                repeatx=1024,   # 可选，防止重复
                                repeaty=1024,   # 可选，防止重复
                                base=base_seed)
            Z[j, i] = noise_val

    # 将噪声值从 [-1, 1] 缩放到期望的高度范围
    Z = Z * height_scale

    return X, Y, Z


def generate_terrain(terrain_data: TerrainData):
    """
    （原始方法）根据三角函数生成山区地形。
    保留此函数作为备选，但在新代码中建议使用柏林噪声。
    """
    # 提取参数
    a = terrain_data.a
    b = terrain_data.b
    c = terrain_data.c
    d = terrain_data.d
    e = terrain_data.e
    f = terrain_data.f
    g = terrain_data.g

    x_min, x_max = terrain_data.x_min, terrain_data.x_max
    y_min, y_max = terrain_data.y_min, terrain_data.y_max
    res = terrain_data.resolution

    # 使用 linspace 确保边界精确
    num_x = int((x_max - x_min) / res) + 1
    num_y = int((y_max - y_min) / res) + 1
    x = np.linspace(x_min, x_max, num_x)
    y = np.linspace(y_min, y_max, num_y)

    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)

    Z = (np.sin(Y + a) +
         b * np.sin(X) +
         c * np.cos(d * r) +
         e * np.cos(Y) +
         f * np.sin(g * r))

    return X, Y, Z