#import tvm.relay.frontend.yolov3 as yolov3
import cv2
import numpy as np
import time

from tvm.relay.frontend.yolo import YOLO

test_image = 'test.jpg'
imagex = cv2.imread(test_image)
imagex = np.array(imagex)

config = {
    'cfg_path': 'yolov3.cfg',
    'weights_path': 'yolov3.weights',
    'device_type': 'cuda-cudnn',
    'autotune': True,
    'log_file': 'yolov3_auto.log',
    'thresh': 0.5,
    'nms_thresh': 0.45
}

# setup
yolov3 = YOLO(config)

# run
start = time.time()
dets = yolov3.run(imagex)
end = time.time()
print(end-start)        # time
print(dets)             # result
