import numpy as np
from util import get_map_coordinates, map_width, map_height


class FilterLidar:
    def interpolate_odometry(current, prev):
        def interpolate(t):
            # print("current.t", current.t, "prev.t", prev.t)
            time_diff = current.t - prev.t
            x = prev.x * (current.t - t) / time_diff + current.x * (t - prev.t) / time_diff
            y = prev.y * (current.t - t) / time_diff + current.y * (t - prev.t) / time_diff
            theta = prev.theta * (current.t - t) / time_diff + current.theta * (t - prev.t) / time_diff

            return x, y, theta
        
        return interpolate
        
    def filter_lidar(lidar, current, prev, likelihood_fields, threshold=None):
        interpolate = FilterLidar.interpolate_odometry(current, prev)
        if threshold is None:
            threshold = 3*likelihood_fields.range_accuracy
        filtered = []
        x, y, theta = interpolate(lidar.t)
        for i in range(lidar.length):
            angle = i * np.pi / 180 + theta -np.pi/2
            lidar_x = x + lidar.data[i] * np.cos(angle)
            lidar_y = y + lidar.data[i] * np.sin(angle)
            lidar_map_x, lidar_map_y = get_map_coordinates(lidar_x, lidar_y)
            if lidar_map_x < 0 or lidar_map_x >= map_width or lidar_map_y < 0 or lidar_map_y >= map_height:
                # filtered.append(None)
                continue
            elif likelihood_fields.likelihood_fields['dmin'][lidar_map_y, lidar_map_x] < threshold:
                filtered.append((lidar_x, lidar_y))
            # else:
            #     filtered.append(None)

        return filtered
    
    def get_pts_from_lidar(lidar_scan_data, x_c, y_c, theta):
        if isinstance(lidar_scan_data, list):
            lidar_scan_data = dict(enumerate(lidar_scan_data))
        pts = []
        for i, v in lidar_scan_data.items():
            angle = i * np.pi / 180 + theta -np.pi/2
            x = x_c + v * np.cos(angle)
            y = y_c + v * np.sin(angle)
            pts.append([x, y])
        return pts