import cv2
from threading import Thread

from utils import save_image, draw_matches, debug


class MatcherThread(Thread):
    def __init__(self, res_1, res_2, name, config):
        super(MatcherThread, self).__init__(name=name)
        self.res_1 = res_1
        self.res_2 = res_2
        self.results = None
        self.config = config

    def run(self):
        matcher_config = self.config.get('matcher')
        # Dettagli di configurazione del matcher.
        index_params = dict(algorithm=matcher_config['algorithm'], trees=matcher_config['trees'])
        search_params = dict(checks=matcher_config['checks'])
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(self.res_1['desc'], self.res_2['desc'], k=2)
        pts_1 = []; pts_2 = []; good = []
        for i, (m, n) in enumerate(matches):
            # Ratio test per filtrare i match ottimali.
            if m.distance < matcher_config['ratio'] * n.distance:
                good.append(m)
                pts_1.append(self.res_1['kp'][m.queryIdx].pt)
                pts_2.append(self.res_2['kp'][m.trainIdx].pt)
        debug("Found {} matches. Good matches are {}" . format(len(matches), len(good)))
        matched_image = draw_matches(self.res_1['img'], self.res_1['kp'], self.res_2['img'], self.res_2['kp'], good)
        save_image(matched_image, self.name, self.res_1['ext'])
        self.results = {'pts_1': pts_1, 'pts_2': pts_2}

    def join(self, timeout=None):
        """
        Override del metodo di join del thread. Viene prima effettuato il join e poi ritornati
        i risultati della computazione del metodo run.
        :param timeout: eventuale timeout per il join.
        :return: i risultati della computaziome.
        """
        super(MatcherThread, self).join(timeout)
        return self.results
