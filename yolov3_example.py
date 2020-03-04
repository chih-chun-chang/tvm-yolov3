import tvm.relay.frontend.yolov3 as yolov3
import cv2
import numpy as np

test_image = 'test.jpg'
imagex = cv2.imread(test_image)
imagex = np.array(imagex)

config = {
    'img': imagex,
    'cfg_path': 'yolov3.cfg',
    'weights_path': 'yolov3.weights',
    'device_type': 'cuda-cudnn',
    'autotune': True,
    'log_file': 'yolov3_auto.log',
    'thresh': 0.5,
    'nms_thresh': 0.45
}

dets = yolov3.run(config)
print(dets)
