import numpy as np
import sys 
import os
import cv2

import tvm 
from tvm import relay
from ctypes import *
from tvm.relay.testing.darknet import __darknetffi__
import tvm.relay.testing.yolo_detection
import tvm.relay.testing.darknet
from tvm import autotvm
from tvm.contrib import graph_runtime
from tvm.contrib.download import download_testdata

import time

class YOLO:
    def  __init__(self, config):
        cfg_path = config['cfg_path']
        weights_path = config['weights_path']
        device_type = config['device_type']
        autotune = config['autotune']
        log_file = config['log_file']
        self.thresh = config['thresh']
        self.nms_thresh = config['nms_thresh']

        DARKNET_URL = 'https://github.com/dmlc/web-data/blob/master/darknet/lib/libdarknet2.0.so?raw=true'
        lib_path = download_testdata(DARKNET_URL, 'libdarknet2.0.so', module="darknet")
        DARKNET_LIB = __darknetffi__.dlopen(lib_path)
        self.net = DARKNET_LIB.load_network(cfg_path.encode('utf-8'), weights_path.encode('utf-8'), 0)
        
        dtype = 'float32'
        data = np.empty([1, self.net.c, self.net.h, self.net.w], dtype)
        self.shape = {'data': data.shape}
        
        # convert darknet to relay functions
        mod, params = relay.frontend.from_darknet(self.net, dtype=dtype, shape=data.shape)

        # import graph to relay
        if device_type == 'cpu':
            target = 'llvm'
            ctx = tvm.cpu(0)
            if autotune:
                if not os.path.isfile(log_file):
                    err = "Autotvm log file does not exist."
                    raise NotImplementedError(err)
                with autotvm.apply_history_best(log_file):
                    with relay.build_config(opt_level=3):
                        graph, lib, params = relay.build_module.build(mod, target=target, params=params)
            else:
                with relay.build_config(opt_level=3):
                    graph, lib, params = relay.build_module.build(mod, target=target, params=params)
        elif device_type == 'cuda-cudnn':
            target = 'cuda -libs=cudnn'
            ctx = tvm.gpu()
            if autotune:
                if not os.path.isfile(log_file):
                    err = "Autotvm log file does not exist."
                    raise NotImplementedError(err)
                with autotvm.apply_history_best(log_file):
                    with relay.build_config(opt_level=3):
                        graph, lib, params = relay.build_module.build(mod, target=target, params=params)
            else:
                with relay.build_config(opt_level=3):
                    graph, lib, params = relay.build_module.build(mod, target=target, params=params)
        elif device_type == 'cuda':
            target = tvm.target.cuda()
            ctx = tvm.gpu()
            if autotune:
                if not os.path.isfile(log_file):
                    err = "Autotvm log file does not exist."
                    raise NotImplementedError(err)
                with autotvm.apply_history_best(log_file):
                    with relay.build_config(opt_level=3):
                        graph, lib, params = relay.build_module.build(mod, target=target, params=params)
            else:
                with relay.build_config(opt_level=3):
                    graph, lib, params = relay.build_module.build(mod, target=target, params=params)
        else:
            err = "Device type is not supported on this platform."
            raise NotImplementedError(err)
        
        self.m = graph_runtime.create(graph, lib, ctx)
        self.m.set_input(**params)


    def run(self, img):
        isinstance(img, np.ndarray)
        [neth, netw] = self.shape['data'][2:] 
        data = tvm.relay.testing.darknet._letterbox_image(img, netw, neth)
        # set inputs
        self.m.set_input('data', tvm.nd.array(data.astype('float32')))
        # execute
        self.m.run()
        # get outputs
        tvm_out = []
        for i in range(3):
            layer_out = {}
            layer_out['type'] = 'Yolo'
            # Get the yolo layer attributes (n, out_c, out_h, out_w, classes, total)
            layer_attr = self.m.get_output(i*4+3).asnumpy()
            layer_out['biases'] = self.m.get_output(i*4+2).asnumpy()
            layer_out['mask'] = self.m.get_output(i*4+1).asnumpy()
            out_shape = (layer_attr[0], layer_attr[1]//layer_attr[0], layer_attr[2], layer_attr[3])
            layer_out['output'] = self.m.get_output(i*4).asnumpy().reshape(out_shape)
            layer_out['classes'] = layer_attr[4]
            tvm_out.append(layer_out)
       
        im_h, im_w, _ = img.shape
        dets = tvm.relay.testing.yolo_detection.fill_network_boxes((netw, neth), (im_w, im_h), self.thresh, 1, tvm_out)
        last_layer = self.net.layers[self.net.n - 1]
        tvm.relay.testing.yolo_detection.do_nms_sort(dets, last_layer.classes, self.nms_thresh)
        results = []
        for det in dets:
            labelstr = []
            category = -1
            for j in range(80):
                if det['prob'][j] > self.thresh:
                    if category == -1:
                        category = j
                    #labelstr.append(j)
                    break
            if category > -1:
                #imc, imh, imw = img.shape
                #classes = labelstr[0]
                classes = category
                b = det['bbox']
                left = int((b.x-b.w/2.)*im_w)
                right = int((b.x+b.w/2.)*im_w)
                top = int((b.y-b.h/2.)*im_h)
                bot = int((b.y+b.h/2.)*im_h)
                if left < 0:
                    left = 0
                if right > im_w-1:
                    right = im_w-1
                if top < 0:
                    top = 0
                if bot > im_h-1:
                    bot = im_h-1
                results.append([classes, left, top, right, bot])
        return results

