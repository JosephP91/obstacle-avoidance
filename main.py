from mops import MOPSThread
from sift import SIFTThread
from match import MatcherThread
from depth import DepthCalcThread
from utils import draw3DPoints, save_image
from config import Configuration

extension = '.png'
image_1 = '1{}' . format(extension)
image_2 = '2{}' . format(extension)

config = Configuration()

mops_task_1 = MOPSThread(image_1, 'MOPS_1', config)
mops_task_2 = MOPSThread(image_2, 'MOPS_2', config)

sift_task_1 = SIFTThread(image_1, 'SIFT_1', config)
sift_task_2 = SIFTThread(image_2, 'SIFT_2', config)

mops_task_1.start()
mops_task_2.start()
sift_task_1.start()
sift_task_2.start()

mops_res_1 = mops_task_1.join()
mops_res_2 = mops_task_2.join()
sift_res_1 = sift_task_1.join()
sift_res_2 = sift_task_2.join()

mops_matcher = MatcherThread(mops_res_1, mops_res_2, 'MOPS_Match', config)
sift_matcher = MatcherThread(sift_res_1, sift_res_2, 'SIFT_Match', config)

mops_matcher.start()
sift_matcher.start()

mops_matches = mops_matcher.join()
sift_matches = sift_matcher.join()

depth_mops_task = DepthCalcThread(mops_matches, 'MOPS_Depth', config)
depth_sift_task = DepthCalcThread(sift_matches, 'SIFT_Depth', config)

depth_mops_task.start()
depth_sift_task.start()

mops_depth_res = depth_mops_task.join()
sift_depth_res = depth_sift_task.join()

image = draw3DPoints(image_2, mops_depth_res + sift_depth_res)
save_image(image, "results", extension)
