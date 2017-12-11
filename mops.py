import cv2
import numpy
from time import time
from math import sin, cos, sqrt

from utils import debug, save_image
from base import FeatureExtractorThread
from corner import CornerExtractorThread


def adaptive_non_maximal_suppression(keypoints, number, robustness):
    suppressed = []
    length = len(keypoints)
    for x in range(0, length):
        # Inizializzazione ad un valore massimo. In questo caso il massimo degli interi.
        radius = numpy.iinfo(numpy.int32)
        xi, yi = keypoints[x].pt[0], keypoints[x].pt[1]
        for y in range(0, length):
            xj, yj = keypoints[y].pt[0], keypoints[y].pt[1]
            if (xi != xj and yi != yj) and keypoints[x].response < robustness * keypoints[y].response:
                dist = sqrt((xj - xi) ** 2 + (yj - yi) ** 2)
                if dist < radius:
                    radius = dist
        suppressed.append([keypoints[x], radius])
    suppressed.sort(key=lambda item: item[1])
    suppressed = suppressed[-number:-1]
    return zip(*suppressed)[0]


def detectAndCompute(image, config):
    """
    Funzione che implementa l'algoritmo MOPS. Dapprima viene generata una piramide gaussiana
    con numero di livelli specificato nel file di configurazione. Avremo quindi una immagine
    scalata per un numero x di livelli. Ad ogni livello della piramide applichiamo l'estrazione
    degli Harris corners, nei quali si troveranno i feature point MOPS. Ogni livello della piramide
    viene processato in maniera concorrente con un thread separato. Una volta terminata questa
    computazione, viene applicato un algoritmo di adaptive non maximal suppression che effettua
    la soppressione dei corner non affidabili, o meglio che hanno una "strength" non elevatissima.
    Al termina di questa procedura avremo un vettore di keypoints nel formato previsto da OpenCV,
    avendo rispettato l'interfaccia cv2.Keypoint. Di conseguenza potremo calcolare dei descrittori
    da utilizzare successivamente.
    :param image: l'immagine sulla quale operare.
    :param config: oggetto Configuration per accedere ai parametri di configurazione.
    :return: i keypoint ed i descriptors calcolati.
    """
    mops, number, levels = config.get('mops'), config.get('points'), config.get('levels')
    keypoints, corners_tasks = [], []

    # Generazione della piramide gaussiana.
    pyramid = [numpy.float32(image)]
    for i in range(1, levels):
        pyramid.append(pyramid[i - 1])
        pyramid[i] = cv2.pyrDown(cv2.GaussianBlur(pyramid[i], (3, 3), 1))
        pyramid[i] = cv2.pyrUp(pyramid[i])

    # Estrazione degli Harris corner da tutti i livelli della piramide.
    for i in range(0, levels):
        task = CornerExtractorThread(pyramid[i], i, keypoints, config)
        corners_tasks.append(task)
        task.start()

    # Attendiamo che i task di estrazione siano terminati.
    for i in range(0, levels):
        corners_tasks[i].join()

    # Applichiamo l'Adaptive Non-Maximal Suppression.
    if mops['use_anms']:
        keypoints = adaptive_non_maximal_suppression(keypoints, number, mops['anms_robustness'])

    # Calcolo dei descriptors dati i keypoints in input.
    return keypoints, cv2.SIFT(number, levels).compute(image, keypoints)[1]


def drawKeypoints(image, keypoints, config):
    """
    Funzione che consente di disegnare i keypoints MOPS sull'immagine considerando anche la loro
    orientazione.
    :param image: l'immagine sulla quale operare.
    :param keypoints: i keypoints da disegnare.
    :param config parametri di configurazione.
    :return: una copia dell'immagine per non alterare quella originale.
    """
    mops = config.get('mops')
    thickness, radius, color = mops['kp_thickness'], mops['kp_radius'], tuple(mops['kp_color'])
    copy = image.copy()
    for keypoint in keypoints:
        y, x = int(keypoint.pt[0]), int(keypoint.pt[1])
        cv2.circle(copy, (y, x), mops['kp_radius'], color, thickness)
        circ_x = numpy.int32(y + sin(keypoint.angle) * radius)
        circ_y = numpy.int32(x + cos(keypoint.angle) * radius)
        cv2.line(copy, (y, x), (circ_x, circ_y), color, thickness)
    return copy


class MOPSThread(FeatureExtractorThread):
    """
    Classe che rappresenta un thread MOPS per il calcolo in maniera concorrente dei descrittori MOPS
    su una immagine specificata in input.
    """
    def __init__(self, image_path, name, config):
        super(MOPSThread, self).__init__(image_path, name)
        self.results = None
        self.config = config

    def run(self):
        start_time = time()
        keypoints, descriptors = detectAndCompute(self.image, self.config)
        debug("MOPS time: {} seconds.".format(time() - start_time))
        self.results = {'img': self.image, 'ext': self.extension, 'kp': keypoints, 'desc': descriptors}
        copy = drawKeypoints(self.image, keypoints, self.config)
        save_image(copy, self.name, self.extension)

    def join(self, timeout=None):
        """
        Override del metodo di joining di un thread. Viene dapprima effettuato il join del thread
        e poi vengono ritornati i risultati della computazione precedente.
        :param timeout: eventuale timeout d'attesa.
        :return: i risultati della computazione precedente.
        """
        super(MOPSThread, self).join(timeout)
        return self.results
