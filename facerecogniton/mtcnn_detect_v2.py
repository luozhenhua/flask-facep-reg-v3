import sys

from MTCNN.Detection.MtcnnDetector import MtcnnDetector
from MTCNN.Detection.detector import Detector
from MTCNN.Detection.fcn_detector import FcnDetector
from MTCNN.Detection.mtcnn_model import P_Net, R_Net, O_Net


test_mode = "onet"
thresh = [0.9, 0.6, 0.7]
min_face_size = 40
stride = 2
slide_window = False
shuffle = False
#vis = True
detectors = [None, None, None]
prefix = ['models/PNet_landmark/PNet', 'models/RNet_landmark/RNet', 'models/ONet_landmark/ONet']
epoch = [18, 14, 16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
PNet = FcnDetector(P_Net, model_path[0])
detectors[0] = PNet
RNet = Detector(R_Net, 24, 1, model_path[1])
detectors[1] = RNet
ONet = Detector(O_Net, 48, 1, model_path[2])
detectors[2] = ONet
mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)



