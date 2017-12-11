import cv2
from time import time
from base import FeatureExtractorThread
from utils import save_image, debug


class SIFTThread(FeatureExtractorThread):
    def __init__(self, image_path, name, config):
        super(SIFTThread, self).__init__(image_path, name)
        self.results = None
        self.config = config

    def run(self):
        sift = cv2.SIFT(self.config.get('points'), self.config.get('levels'))
        start_time = time()
        keypoints, descriptors = sift.detectAndCompute(self.image, None)
        debug("SIFT time: {} seconds.".format(time() - start_time))
        self.results = {'img': self.image, 'ext': self.extension, 'kp': keypoints, 'desc': descriptors}
        image = cv2.drawKeypoints(self.image, keypoints)
        save_image(image, self.name, self.extension)

    def join(self, timeout=None):
        """
        Override del metodo join, che effettua dapprima il joining del thread e poi ritorna i risultati
        della computazione precedente.
        :param timeout: eventuale timeout di connessione.
        :return: i risultati della computazione.
        """
        super(SIFTThread, self).join(timeout)
        return self.results
