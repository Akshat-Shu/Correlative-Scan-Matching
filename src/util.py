import numpy as np
# import cupy as cp
import matplotlib.pyplot as plt

map_file = '../assets/aces_relation_map2.png'

def show_pts(x_data, y_data=None):
    if y_data is None:
        x_data_n, y_data = [], []
        for x in x_data:
            x_data_n.append(x[0])
            y_data.append(x[1])
        x_data = x_data_n
    plt.imshow(plt.imread(map_file))
    plt.scatter(x_data, y_data, c='r', s=1)
    plt.show()

class Odom:
    def __init__(self, x, y, theta, tv=None, rv=None, acc=None, t=None, host=None, dt=None):
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta)
        self.host = host

        if tv is not None:
            self.tv = float(tv)
        else: self.tv = None
        if rv is not None:
            self.rv = float(rv)
        else: self.rv = None
        if acc is not None:
            self.acc = float(acc)
        else: self.acc = None
        if t is not None:
            self.t = float(t)
        else:
            self.t = None
        if dt is not None:
            self.dt = float(dt)
        else: self.dt = None

    def __sub__(self, other):
        return self.x - other.x, self.y - other.y, self.theta - other.theta

    def __str__(self):
        return f"x: {self.x}, y: {self.y}, theta: {self.theta}, tv: {self.tv}, rv: {self.rv}, acc: {self.acc}, t: {self.t}, host: {self.host}, dt: {self.dt}"

class Lidar:
    def __init__(self, length, data, x, y, theta, odom_x, odom_y, odom_theta, t, host, dt):
        self.length = int(length)
        if isinstance(data, list):
            dat = {}
            for i in range(length):
                dat[i] = data[i]
            self.data = dat
        else:
            self.data = data
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta)
        self.odom_x = float(odom_x)
        self.odom_y = float(odom_y)
        self.odom_theta = float(odom_theta)
        self.t = float(t)
        self.host = host
        self.dt = float(dt)


def get_map_coordinates(x, y):
    ax, bx = 10, 316
    ay, by = 10, 139

    return (int(-y*ax+bx), int(-x*ay+by))

def get_map_coordinates_list(x_l, y_l):
    # if isinstance(x_l, cp.ndarray):
    #     x_l = x_l.get()
    # if isinstance(y_l, cp.ndarray):
    #     y_l = y_l.get()
    ax, bx = 10, 316
    ay, by = 10, 139

    return (np.asarray(-y_l*ax+bx, dtype=int), np.asarray(-x_l*ay+by, dtype=int))



def deep_convert(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {deep_convert(key): deep_convert(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [deep_convert(element) for element in obj]
    elif isinstance(obj, tuple):
        return tuple(deep_convert(element) for element in obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    return obj


def get_data(file):
    with open(file, 'r') as f:
        data = f.readlines()

    def parse_odom(s):
        s = s.split()
        if s[0] == 'ODOM':
            s = s[1:]
        
        return Odom(*s)
    
    def parse_lidar(s):
        s = s.split()
        if s[0] == 'FLASER':
            s = s[1:]
        
        length = int(s[0])
        s = s[1:]
        data = [float(x) for x in s[:length]]
        return Lidar(length, data, *s[length:])
        
    
    data = [x.strip() for x in data[13:]]
    odoms, lidars = [], []
    for i in range(len(data)):
        if data[i].startswith('ODOM'):
            odoms.append(parse_odom(data[i]))
        elif data[i].startswith('FLASER'):
            lidars.append(parse_lidar(data[i]))

    return odoms, lidars

map_width = 669
map_height = 662