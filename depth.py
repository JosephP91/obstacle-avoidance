import cv2
import numpy as np
import math
from threading import Thread
from time import time

from utils import debug


class DepthCalcThread(Thread):
    def __init__(self, matches, name, config):
        super(DepthCalcThread, self).__init__(name=name)
        self.pts_1 = matches['pts_1']
        self.pts_2 = matches['pts_2']
        self.config = config
        self.results = []

    def run(self):
        distances = []
        length = len(self.pts_1)
        start_time = time()
        for i in range(0, length):
            x_1 = self.pts_1[i][0]
            y_1 = self.pts_1[i][1]
            x_2 = self.pts_2[i][0]
            y_2 = self.pts_2[i][1]
            distance = math.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)
            distances.append(distance)
        debug("Depth computation time: {} seconds." . format(time() - start_time))
        if self.config.get('normalize_points'):
            distances = cv2.normalize(np.array(distances), 0, 255, norm_type=cv2.NORM_MINMAX)
        for i in range(0, length):
            self.results.append((int(self.pts_2[i][0]), int(self.pts_2[i][1]), round(distances[i], 2)))
        if self.config.get('sort_points'):
            self.results.sort(key=lambda x: x[2])

    def join(self, timeout=None):
        super(DepthCalcThread, self).join(timeout)
        return self.results
