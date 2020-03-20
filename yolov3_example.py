import cv2
import numpy as np
import time

from tvm.relay.frontend.yolo import YOLO

test_image = 'frame_1.jpg'
imagex = cv2.imread(test_image)


config = {
    'cfg_path': 'yolov3.cfg',
    'weights_path': 'yolov3.weights',
    'device_type': 'cuda-cudnn',
    'autotune': True,
    'log_file': 'yolov3_auto.log',
    'thresh': 0.3,
    'nms_thresh': 0.4
}

# setup
yolov3 = YOLO(config)

# run
start = time.time()
#for _ in range(100):
dets = yolov3.run(imagex)
end = time.time()
print((end-start)/100)        # time
print(dets)             # result
