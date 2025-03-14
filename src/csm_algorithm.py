import numpy as np
from util import get_map_coordinates
from filter_lidar import FilterLidar
from util import Lidar
from util import map_height, map_width

def predict_pose(previous_pose, odometry_delta):
    x, y, theta = previous_pose.x, previous_pose.y, previous_pose.theta
    dx, dy, dtheta = odometry_delta
    
    new_pose = previous_pose.get_next_pose(dx, dy, dtheta)
    return new_pose


def fuse_pose(neighbours):
    weights = np.array([n[3] for n in neighbours])
    weights /= np.sum(weights)
    fused_pose = np.sum(np.array([n[:3] for n in neighbours]) * weights[:, None], axis=0)
    return fused_pose

def apply_low_pass_filter(corrected_pose, predicted_pose, kp=0.1, ke=10000):
    covariance_matrix = np.cov(np.array(corrected_pose[:2]), np.array(predicted_pose[:2]))
    eigenvalues = np.linalg.eigvals(covariance_matrix[:2, :2])
    E = np.diag(eigenvalues)
    filtered_pose = np.linalg.inv((1 + kp) * np.eye(2) + ke * E) @ (corrected_pose[:2] + (kp * np.eye(2) + ke * E) @ predicted_pose[:2])
    return np.array([filtered_pose[0], filtered_pose[1], corrected_pose[2]])

def matching_2d_slices(scores):
    poses = list(scores.keys())
    best_pose = min(scores.keys(), key=scores.get)
    neighbors = sorted(poses, key=lambda p: np.linalg.norm(np.array([p[0], p[1], p[2], scores[(p)]]) - np.array([best_pose[0], best_pose[1], best_pose[2], scores[(best_pose)]])))[:4]
    return fuse_pose([[n[0], n[1], n[2], scores[n]] for n in neighbors])

def get_candidates(lidar, likelihood_fields, odom, prev_odom, search_space, fine, lidar_scatter=None):
    angle_candidates = np.linspace(-np.pi/27, np.pi/27, 31)
    Vs = {}
    scores = {}
    pose_scores = {}

    for j, angle in enumerate(angle_candidates):
        n_theta = odom.theta + angle
        prev_odom.theta = prev_odom.theta + angle
        odom.theta = odom.theta + angle
        n_lidar = Lidar(lidar.length, lidar.data, lidar.x, lidar.y, lidar.theta, lidar.odom_x, lidar.odom_y, lidar.odom_theta, lidar.t, lidar.host, lidar.dt)

        Vs[j] = FilterLidar.filter_lidar(n_lidar, odom, prev_odom, likelihood_fields, 5)
        prev_odom.theta = prev_odom.theta - angle
        odom.theta = odom.theta - angle

    for i, angle in enumerate(angle_candidates):
        for x in np.linspace(-search_space, search_space, fine):
            for y in np.linspace(-search_space, search_space, fine):
                dx, dy = x, y
                # dx, dy = 0, 0
                dtheta = angle
                n_pose = odom.get_next_pose(dx, dy, dtheta)
                
                num = len(Vs[i])
                if(num == 0): continue
                x_data, y_data = [], []
                sum = 0
                for pt in Vs[i]:
                    xd, yd = pt
                    # print(f"xd: {xd}, yd: {yd}")
                    map_x, map_y = get_map_coordinates(xd+dx, yd+dy)
                    x_data.append(map_x)
                    y_data.append(map_y)
                    if map_x < 0 or map_x >= map_width or map_y < 0 or map_y >= map_height:
                        num -= 1
                        continue
                    sum += likelihood_fields.likelihood_fields['score'][map_y, map_x]
                scores[(dx, dy, angle)] = sum / num
                pose_scores[(n_pose.x, n_pose.y, n_pose.theta)] = scores[(dx, dy, angle)]

    x_data, y_data = [], []
    best_x, best_y, best_theta = min(scores.keys(), key=scores.get)
    if lidar_scatter:
        for pt in Vs[angle_candidates.tolist().index(best_theta)]:
            xd, yd = pt
            map_x, map_y = get_map_coordinates(xd+best_x, yd+best_y)
            x_data.append(map_x)
            y_data.append(map_y)
        lidar_scatter.set_offsets(np.c_[x_data, y_data])
    return scores, Vs, pose_scores



import numpy as np
from util import get_map_coordinates_list

# Taking the simple implementation and vectorizing it for performance

def get_candidates_np(lidar, likelihood_fields, odom, prev_odom, search_space, fine, lidar_scatter=None):
    angle_candidates = np.linspace(-np.pi/27, np.pi/27, 31)  # 31 angle candidates
    scores = {}
    n_pts = {}
    pose_scores = {}
    Vs = {}

    for j, angle in enumerate(angle_candidates):
        n_theta = odom.theta + angle
        prev_odom.theta = prev_odom.theta + angle
        odom.theta = odom.theta + angle
        n_lidar = Lidar(lidar.length, lidar.data, lidar.x, lidar.y, lidar.theta, lidar.odom_x, lidar.odom_y, lidar.odom_theta, lidar.t, lidar.host, lidar.dt)
        Vs[j] = FilterLidar.filter_lidar(n_lidar, odom, prev_odom, likelihood_fields)
        prev_odom.theta = prev_odom.theta - angle
        odom.theta = odom.theta - angle

    dx_values = np.linspace(-search_space, search_space, fine)
    dy_values = np.linspace(-search_space, search_space, fine)

    dx_grid, dy_grid, angle_grid = np.meshgrid(dx_values, dy_values, angle_candidates, indexing='ij')

    score_matrix = np.full(dx_grid.shape, np.inf)

    for i, angle in enumerate(angle_candidates):
        if len(Vs[i]) == 0:
            continue

        lidar_x, lidar_y = np.array(Vs[i]).T 

        transformed_x = lidar_x[:, None, None] + dx_grid[:, :, i]
        transformed_y = lidar_y[:, None, None] + dy_grid[:, :, i]

        map_x, map_y = get_map_coordinates_list(transformed_x, transformed_y)

        valid_mask = (map_x >= 0) & (map_x < map_width) & (map_y >= 0) & (map_y < map_height)

        likelihood_values = np.zeros_like(map_x, dtype=np.float32)
        likelihood_values[valid_mask] = likelihood_fields.likelihood_fields['score'][map_y[valid_mask], map_x[valid_mask]]

        score_matrix[:, :, i] = np.sum(likelihood_values, axis=0) / (np.sum(valid_mask, axis=0) + 1e-8)  # Avoid division by zero

    best_idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
    best_x, best_y, best_theta = dx_values[best_idx[0]], dy_values[best_idx[1]], angle_candidates[best_idx[2]]

    
    for i in range(dx_values.shape[0]):
        for j in range(dy_values.shape[0]):
            for k in range(angle_candidates.shape[0]):
                scores[(dx_values[i], dy_values[j], angle_candidates[k])] = score_matrix[i, j, k]
                n_pts[(dx_values[i], dy_values[j], angle_candidates[k])] = len(Vs[k])

    if lidar_scatter:
        best_pts = Vs[best_idx[2]]
        best_x_data, best_y_data = np.array(best_pts).T
        x_data, y_data = get_map_coordinates_list(best_x_data + best_x, best_y_data + best_y)
        lidar_scatter.set_offsets(np.c_[x_data, y_data])

    return scores, Vs, pose_scores, n_pts
