import cv2 as cv
import numpy as np

class LikelihoodFields:
    def __init__(self, occupancy_map, max_search_dist, range_accuracy):
        if len(occupancy_map.shape) != 2:
            occupancy_map = cv.cvtColor(occupancy_map, cv.COLOR_BGR2GRAY)
        self.occupancy_map = np.asarray(occupancy_map)
        self.height, self.width = occupancy_map.shape

        self.likelihood_fields = np.zeros((self.height, self.width), dtype=[('dmin', float), ('score', float)])
        self.max_search_dist = max_search_dist
        self.range_accuracy = range_accuracy

        self.lowest_score = self.calc_measurement_model(max_search_dist)
        self.score_for_unknown = self.lowest_score / 10

        self.compute_likelihood_fields()

    def compute_likelihood_fields(self):
        binary_map = np.where(self.occupancy_map == 0, 0, 1).astype(np.uint8)

        dist_transform = cv.distanceTransform(binary_map, distanceType=cv.DIST_L2, maskSize=5)

        self.likelihood_fields['dmin'] = dist_transform
        self.likelihood_fields['score'] = self.calc_measurement_model(dist_transform)

        # # Assign predefined values for occupied and unknown cells
        self.likelihood_fields['dmin'][self.occupancy_map == 0] = 0
        self.likelihood_fields['score'][self.occupancy_map == 0] = 1
        self.likelihood_fields['dmin'][self.occupancy_map == 255] = self.max_search_dist
        self.likelihood_fields['score'][self.occupancy_map == 255] = self.score_for_unknown

    def calc_measurement_model(self, d):
        return np.exp(-0.5 * (d / self.range_accuracy) ** 2)
