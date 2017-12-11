from cv2 import cornerHarris, KeyPoint, Sobel, CV_32F, phase
from threading import Thread


class CornerExtractorThread(Thread):
    def __init__(self, image, level, keypoints, config):
        super(CornerExtractorThread, self).__init__()
        self.keypoints = keypoints
        self.image = image
        self.level = level
        self.config = config

    def run(self):
        # Estrazione degli Harris Corner.
        corners = cornerHarris(self.image, blockSize=2, ksize=3, k=0.04)
        # Calcolo orientamento keypoints tramite gradienti e ad angolo tra questi ultimi.
        grad_x = Sobel(self.image, CV_32F, 1, 0, ksize=3, scale=4.5)
        grad_y = Sobel(self.image, CV_32F, 0, 1, ksize=3, scale=4.5)
        theta = phase(grad_x, grad_y, angleInDegrees=True)
        # Filtraggio e creazione degli oggetti Keypoint di OpenCV.
        strength = int(self.config.get('mops')['corner_ratio'] * corners.max())
        for x in range(0, self.image.shape[0]):
            for y in range(0, self.image.shape[1]):
                if int(corners[x][y]) > strength:
                    self.keypoints.append(
                        KeyPoint(y, x, _size=3,  _angle=theta[x][y], _response=corners[x][y], _octave=self.level))
