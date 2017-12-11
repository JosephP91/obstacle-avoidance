import cv2
from os.path import splitext, isfile
from os import getcwd
from threading import Thread
from utils import debug


class FeatureExtractorThread(Thread):
    def __init__(self, image_path, name):
        super(FeatureExtractorThread, self).__init__(name=name)
        fullpath = getcwd() + "/images/" + image_path
        if not isfile(fullpath):
            debug("Invalid file specified {}".format(fullpath))
            exit()
        self.image = cv2.imread(fullpath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        self.extension = splitext(image_path)[1]
