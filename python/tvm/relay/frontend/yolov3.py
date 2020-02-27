# Compile yolov3 for demo

import numpy as np
import sys
import os

import tvm
from tvm import relay
from ctypes import *
from tvm.relay.testing.darknet import __darknetffi__
import tvm.relay.testing.yolo_detection
import tvm.relay.testing.darknet
from tvm import autotvm
from tvm.contrib import graph_runtime
from tvm.contrib.download import download_testdata

def run(config):
    img_path = config['image_path']
    cfg_path = config['cfg_path']
    weights_path = config['weights_path']
    device_type = config['device_type']
    autotune = config['autotune']
    log_file = config['log_file']
    thresh = config['thresh']
    nms_thresh = config['nms_thresh']

    REPO_URL = 'https://github.com/dmlc/web-data/blob/master/darknet/'
    if sys.platform in ['linux', 'linux2']:
        DARKNET_LIB = 'libdarknet2.0.so'
        DARKNET_URL = REPO_URL + 'lib/' + DARKNET_LIB + '?raw=true'
    else:
        err = "Darknet lib is not supported on {} platform".format(sys.platform)
        raise NotImplementedError(err)
	
    lib_path = download_testdata(DARKNET_URL, DARKNET_LIB, module="darknet")
    DARKNET_LIB = __darknetffi__.dlopen(lib_path)
    net = DARKNET_LIB.load_network(cfg_path.encode('utf-8'), weights_path.encode('utf-8'), 0)
	
    dtype = 'float32'

    data = np.empty([1, net.c, net.h, net.w], dtype)
    shape = {'data': data.shape}
    print("Converting darknet to relay functions...")
    mod, params = relay.frontend.from_darknet(net, dtype=dtype, shape=data.shape)

    print("Import the graph to Relay...")
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
    
    print("Loading image...")
    [neth, netw] = shape['data'][2:] # Current image shape is 608x608
    data = tvm.relay.testing.darknet.load_image(img_path, netw, neth)
    m = graph_runtime.create(graph, lib, ctx)

	# set inputs
    m.set_input('data', tvm.nd.array(data.astype(dtype)))
    m.set_input(**params)
	# execute
    print("Running the test image...")

    # detection
    # thresholds

    m.run()
	
    # get outputs
    tvm_out = []
    for i in range(3):
        layer_out = {}
        layer_out['type'] = 'Yolo'
        # Get the yolo layer attributes (n, out_c, out_h, out_w, classes, total)
        layer_attr = m.get_output(i*4+3).asnumpy()
        layer_out['biases'] = m.get_output(i*4+2).asnumpy()
        layer_out['mask'] = m.get_output(i*4+1).asnumpy()
        out_shape = (layer_attr[0], layer_attr[1]//layer_attr[0],
                     layer_attr[2], layer_attr[3])
        layer_out['output'] = m.get_output(i*4).asnumpy().reshape(out_shape)
        layer_out['classes'] = layer_attr[4]
        tvm_out.append(layer_out)

    # do the detection and bring up the bounding boxes
    img = tvm.relay.testing.darknet.load_image_color(img_path)

    _, im_h, im_w = img.shape
    dets = tvm.relay.testing.yolo_detection.fill_network_boxes((netw, neth), (im_w, im_h), thresh, 1, tvm_out)
    last_layer = net.layers[net.n - 1]

    tvm.relay.testing.yolo_detection.do_nms_sort(dets, last_layer.classes, nms_thresh)

    results = []
    for det in dets:
        labelstr = []
        category = -1
        for j in range(80):
            if det['prob'][j] > thresh:
                if category == -1: 
                    category = j 
                #labelstr.append(names[j] + " " + str(round(det['prob'][j], 4)))
        if category > -1: 
            imc, imh, imw = img.shape
            #width = int(imh * 0.006)
            #offset = category*123457 % classes
            #red = _get_color(2, offset, classes)
            #green = _get_color(1, offset, classes)
            #blue = _get_color(0, offset, classes)
            #rgb = [red, green, blue]
            b = det['bbox']
            left = int((b.x-b.w/2.)*imw)
            right = int((b.x+b.w/2.)*imw)
            top = int((b.y-b.h/2.)*imh)
            bot = int((b.y+b.h/2.)*imh)

            if left < 0:
                left = 0 
            if right > imw-1:
                right = imw-1
            if top < 0:
                top = 0 
            if bot > imh-1:
                bot = imh-1
            #_draw_box_width(im, left, top, right, bot, width, red, green, blue)
            #label = _get_label(font_path, ''.join(labelstr), rgb)
            #_draw_label(im, top + width, left, label, rgb)
            r = dict()
            r['left'] = left
            r['right'] = right
            r['top'] = top
            r['bot'] = bot
            results.append(r)
    return results
