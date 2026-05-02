"""
全局路径规划模块：A* 算法 + 大步长重采样 + 高度赋值
"""
import numpy as np
import heapq
from typing import List, Tuple, Optional

class GridMap:
    def __init__(self, x_min, x_max, y_min, y_max, resolution, obstacles, terrain_func=None):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.resolution = resolution
        self.nx = int((x_max - x_min) / resolution) + 1
        self.ny = int((y_max - y_min) / resolution) + 1
        self.obstacles = obstacles  # list of (cx, cy, radius)
        self.terrain_func = terrain_func
        self.cost_map = np.ones((self.nx, self.ny))
        self._init_cost_map()

    def _init_cost_map(self):
        for ox, oy, r in self.obstacles:
            for i in range(self.nx):
                x = self.x_min + i * self.resolution
                for j in range(self.ny):
                    y = self.y_min + j * self.resolution
                    if (x - ox)**2 + (y - oy)**2 <= r**2:
                        self.cost_map[i, j] = np.inf

    def xy_to_index(self, x, y):
        i = int(round((x - self.x_min) / self.resolution))
        j = int(round((y - self.y_min) / self.resolution))
        i = max(0, min(i, self.nx-1))
        j = max(0, min(j, self.ny-1))
        return i, j

    def index_to_xy(self, i, j):
        return self.x_min + i * self.resolution, self.y_min + j * self.resolution

def heuristic(a, b):
    return np.hypot(a[0]-b[0], a[1]-b[1])

def astar_planning(grid_map, start_xy, goal_xy):
    start_idx = grid_map.xy_to_index(start_xy[0], start_xy[1])
    goal_idx = grid_map.xy_to_index(goal_xy[0], goal_xy[1])
    open_set = [(0, start_idx)]
    came_from = {}
    g_score = {start_idx: 0}
    f_score = {start_idx: heuristic(start_xy, goal_xy)}
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal_idx:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_idx)
            path.reverse()
            waypoints = [grid_map.index_to_xy(i, j) for (i, j) in path]
            return waypoints
        cx, cy = grid_map.index_to_xy(current[0], current[1])
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            neighbor = (current[0]+dx, current[1]+dy)
            if 0 <= neighbor[0] < grid_map.nx and 0 <= neighbor[1] < grid_map.ny:
                if grid_map.cost_map[neighbor[0], neighbor[1]] == np.inf:
                    continue
                step_cost = np.hypot(dx*grid_map.resolution, dy*grid_map.resolution)
                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score.get(neighbor, np.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(grid_map.index_to_xy(neighbor[0], neighbor[1]), goal_xy)
                    heapq.heappush(open_set, (f, neighbor))
    return None

def simplify_path(path_xy, step_size):
    if not path_xy:
        return []
    simplified = [path_xy[0]]
    accum = 0.0
    for i in range(1, len(path_xy)):
        seg_len = np.hypot(path_xy[i][0]-path_xy[i-1][0], path_xy[i][1]-path_xy[i-1][1])
        accum += seg_len
        if accum >= step_size:
            simplified.append(path_xy[i])
            accum = 0.0
    if simplified[-1] != path_xy[-1]:
        simplified.append(path_xy[-1])
    return simplified

def assign_altitude(waypoints_xy, terrain_func, safe_height):
    waypoints_3d = []
    for (x, y) in waypoints_xy:
        z = terrain_func(x, y) + safe_height
        waypoints_3d.append([x, y, z])
    return np.array(waypoints_3d)