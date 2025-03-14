import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
from likelihood_fields import LikelihoodFields
import numpy as np
from filter_lidar import FilterLidar
from pose import Pose
from util import Odom, Lidar
from csm_algorithm import get_candidates, matching_2d_slices, apply_low_pass_filter, get_candidates_np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from matplotlib.animation import FuncAnimation
from util import get_map_coordinates, deep_convert, get_data
from global_localization import required, localize


file = '../assets/aces.clf'
img_file = '../assets/aces_relations.png'

img_file = '../assets/aces_relation_map2.png'

img = cv.imread(img_file)


odoms, lidars = get_data(file)

def plot_odoms(odom, img):
    plt.figure(figsize=(10, 10))
    img = plt.imread(img)
    x = [o.x for o in odom]
    y = [o.y for o in odom]

    fig, ax = plt.subplots()
    ax.imshow(img, extent=[min(x), max(x), min(y), max(y)])
    line, = ax.plot([], [], 'ro')

    line.set_data(x, y)
    plt.show()

def extract_guide_lines(img):
    im = cv.imread(img)
    green_color = np.array([0, 153, 0])

    mask = cv.inRange(im, green_color, green_color)
    return mask
 

def extract_unknown_path(img):
    unknown_color = (240, 220, 220)

    im = cv.imread(img)
    mask = cv.inRange(im, unknown_color, unknown_color)
    return mask

# path = extract_guide_lines(img_file)
# cv.imshow('path', path)
# cv.waitKey(0)

# points_file = '../assets/points.txt'
# lines = open(points_file, 'r').readlines()
# points = []
# for l in lines:
#     p = l.split()
#     points.append((float(p[0]), float(p[1])))


# def plot_points(points, img):
#     plt.figure(figsize=(10, 10))
#     img = plt.imread(img)
#     x = [p[0] for p in points]
#     y = [p[1] for p in points]

#     fig, ax = plt.subplots()
#     ax.imshow(img, extent=[min(x), max(x), min(y), max(y)])
#     line, = ax.plot([], [], 'ro')

#     line.set_data(x, y)
#     plt.show()

map_file = '../assets/aces_relation_map2.png'

def get_obstacles(img):
    im = cv.imread(img)
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im = cv.threshold(im, 127, 255, cv.THRESH_BINARY)[1]
    return im

unknown_mask = extract_unknown_path(map_file)

obstacles = get_obstacles(map_file)

def get_occupancy_map(obstacles):
        return obstacles // 255

occupancy_map = get_occupancy_map(obstacles)
occupancy_map[unknown_mask == 255] = 255

likelihood_fields = LikelihoodFields(occupancy_map, 20, 2)
scores = likelihood_fields.likelihood_fields['score']

# plt.figure(figsize=(10, 10))
# plt.imshow(scores, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.title('Likelihood Fields')
# plt.show()


pts = []

fig, axis = plt.subplots()
axis.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
scatter = axis.scatter([], [], c='b', linewidths=0.1, edgecolors='r', s=1)
lidar_scatter = axis.scatter([], [], c='g', s=1)
axis.set_xlim(0, img.shape[1])
axis.set_ylim(img.shape[0], 0)



def __main__():
    init_frame = 0
    prev_odom = odoms[init_frame]
    init_x, init_y, init_theta = 0, 0, -np.pi/2
    global pts
    corrected_odoms = [Pose(0, 0, -np.pi/2)]
    previous_pose = Pose(init_x, init_y, init_theta, t=0)
    
    def init_func():
        pass

    def update(i):
        nonlocal prev_odom, previous_pose
        if(i >= len(lidars)): return
        lidar = lidars[i]
        new_odom = odoms[i+1]
        dx, dy, dtheta = new_odom - prev_odom
        previous_pose.t = prev_odom.t
        dtheta = new_odom.theta - previous_pose.theta
        predicted_pose = previous_pose.get_next_pose(dx, dy, dtheta)
        predicted_pose.t = new_odom.t


        filtered_lidar = FilterLidar.filter_lidar(lidar, predicted_pose, previous_pose, likelihood_fields)
        interp_x, interp_y, interp_theta = FilterLidar.interpolate_odometry(new_odom, prev_odom)(lidar.t)
        scores, Vs, pose_scores, n_pts = get_candidates_np(lidar, likelihood_fields, 
                                predicted_pose, previous_pose, 0.06, 41, lidar_scatter)
        dx, dy, dtheta = max(scores.keys(), key=scores.get)
        removal_ratio = (180-n_pts[(dx, dy, dtheta)])/180
        new_pose = predicted_pose
        if required(likelihood_fields, predicted_pose, removal_ratio, lidar):
            print("localizing")
            scores, Vs, pose_scores = localize(lidar, likelihood_fields, predicted_pose, previous_pose, lidar_scatter)
            corrected_pose = max(pose_scores.keys(), key=pose_scores.get)
            corrected_pose = Pose(corrected_pose[0], corrected_pose[1], corrected_pose[2], t=new_odom.t)
            new_pose = corrected_pose
        else:
            corrected_pose = predicted_pose
            corrected_pose.t = new_odom.t 
            corrected_pose.x += dx
            corrected_pose.y += dy
            corrected_pose.theta += dtheta      
            # corrected_pose = matching_2d_slices(pose_scores)
            corrected_pose = apply_low_pass_filter(corrected_pose, [predicted_pose.x, predicted_pose.y, predicted_pose.theta])
            corrected_pose = Pose(corrected_pose[0], corrected_pose[1], corrected_pose[2], t=new_odom.t)
            new_pose = corrected_pose
            if i == 1:
                filter_scores = {}
                for k, v in scores.items():
                    if(k[0] == 0 and k[1] == 0):
                        filter_scores[tuple(np.array(k).tolist())] = v.tolist()

                print(filter_scores)

        str = f"new_pose:  x:  {new_pose.x} y:  {new_pose.y} theta:  {new_pose.theta} frame:  {i}"
        print(str)
        with open('poses.txt', 'a') as f:
            f.write(str + '\n')
        corrected_odoms.append(new_pose)
        previous_pose = new_pose
        prev_odom = new_odom

        pts.append(get_map_coordinates(new_pose.x, new_pose.y))

        x_data = [pt[0] for pt in pts]
        y_data = [pt[1] for pt in pts]
        scatter.set_offsets(np.c_[x_data, y_data])
        return scatter, lidar_scatter

    animation = FuncAnimation(fig, update, init_func=init_func, frames=range(init_frame, len(lidars)), interval=1, blit=False)
    plt.show()



if __name__ == '__main__':
    __main__()