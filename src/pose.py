import numpy as np

class Pose:
    def __init__(self, x, y, theta, t=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.t = t

    def get_next_pose(self, dx, dy, dtheta):
        new_theta = self.theta + dtheta
        # print(f"self.theta: {self.theta}")
        new_x = self.x + dx * np.cos(dtheta) - dy * np.sin(dtheta)
        new_y = self.y + dx * np.sin(dtheta) + dy * np.cos(dtheta)
        # new_x = self.x + dx
        # new_y = self.y + dy

        return Pose(new_x, new_y, new_theta, self.t)
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(4))]
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        elif key == 2:
            return self.theta
        elif key == 3:
            return self.t
        else:
            raise IndexError("Index out of range")

    def __setitem__(self, key, value):
        if key == 0:
            self.x = value
        elif key == 1:
            self.y = value
        elif key == 2:
            self.theta = value
        elif key == 3:
            self.t = value
        else:
            raise IndexError("Index out of range")