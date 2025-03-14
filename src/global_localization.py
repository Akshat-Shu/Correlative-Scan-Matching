import numpy as np
from util import get_map_coordinates
from filter_lidar import FilterLidar
from util import Lidar
from util import map_height, map_width
from pose import Pose
import time
import matplotlib.pyplot as plt
from util import get_map_coordinates_list, show_pts

def required(likelihood_field, pose, removal_ratio, lidar, removal_thres=0.8888):
    # Only implemented Global localization but avoid using it since it fails to run on my machine
    return False
    if removal_ratio > removal_thres:
        return True


def localize(lidar, likelihood_fields, odom, prev_odom, lidar_scatter=None):
    angle_candidates = np.linspace(-np.pi, np.pi, 721)
    scores = {}
    pose_scores = {}
    Vs = {}
    search_space = 2.0
    fine = 101

    interp_x, interp_y, interp_theta = FilterLidar.interpolate_odometry(odom, prev_odom)(lidar.t)
    for j, angle in enumerate(angle_candidates):
        Vs[j] = FilterLidar.get_pts_from_lidar(lidar.data, interp_x, interp_y, interp_theta+angle)

    dx_values = np.linspace(-search_space, search_space, fine)
    dy_values = np.linspace(-search_space, search_space, fine)

    dx_grid, dy_grid, angle_grid = np.meshgrid(dx_values, dy_values, angle_candidates, indexing='ij')


    score_matrix = np.full(dx_grid.shape, np.inf)

    for i, angle in enumerate(angle_candidates):
        if len(Vs[i]) == 0:
            continue

        lidar_x, lidar_y = np.array(Vs[i]).T  # Shape: (N,)

        transformed_x = lidar_x[:, None, None] + dx_grid[:, :, i]
        transformed_y = lidar_y[:, None, None] + dy_grid[:, :, i]
        map_x, map_y = get_map_coordinates_list(transformed_x, transformed_y)

        valid_mask = (map_x >= 0) & (map_x < map_width) & (map_y >= 0) & (map_y < map_height)

        likelihood_values = np.zeros_like(map_x, dtype=np.float32)
        likelihood_values[valid_mask] = likelihood_fields.likelihood_fields['score'][map_y[valid_mask], map_x[valid_mask]]

        score_matrix[:, :, i] = np.sum(likelihood_values, axis=0) / (np.sum(valid_mask, axis=0) + 1e-8)  # Avoid division by zero

    best_idx = np.unravel_index(np.argmin(score_matrix), score_matrix.shape)
    best_x, best_y, best_theta = dx_values[best_idx[0]], dy_values[best_idx[1]], angle_candidates[best_idx[2]]

    for i in range(dx_values.shape[0]):
        for j in range(dy_values.shape[0]):
            for k in range(angle_candidates.shape[0]):
                scores[(dx_values[i], dy_values[j], angle_candidates[k])] = score_matrix[i, j, k]
                pose_scores[(interp_x + dx_values[i], interp_y + dy_values[j], interp_theta + angle_candidates[k])] = score_matrix[i, j, k]

    if lidar_scatter:
        best_pts = Vs[best_idx[2]]
        best_x_data, best_y_data = np.array(best_pts).T
        x_data, y_data = get_map_coordinates_list(best_x_data + best_x, best_y_data + best_y)
        lidar_scatter.set_offsets(np.c_[x_data, y_data])


    return scores, Vs, pose_scores
