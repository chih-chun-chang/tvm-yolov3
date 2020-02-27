import tvm.relay.frontend.yolov3 as yolov3

config = {
    'image_path': 'test.jpg',
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
