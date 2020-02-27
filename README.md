# Compile darknet on tvm

This is a demo of yolov3 on TVM. 

## Environments Setup

1. **Install TVM**

    1. Requirements
    
    ```
    sudo apt-get update 
    sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
    ```
    
    2. Download llvm pre-built binary from [here](http://releases.llvm.org/download.html) (depends on your OS)
    
    > unzip llvm directory under `tvm-yolov3/`
    
    3. Compile (modify `build/cmake.config` if needed)
    
    ```
    cd build/ && cmake ..
    make -j8
    ```
    
    4. Install Python required packages
    
    `pip install -r requirements.txt`
    
    ###### for other TVM intallation issues please refer to the [website](https://docs.tvm.ai/install/from_source.html)
    
2. **Prepare Data**

    1. Download yolov3 [weights](https://pjreddie.com/media/files/yolov3.weights) and unzip it under `tvm-yolov3/`

## Run and Testing

`import tvm.relay.frontend.yolov3 as yolov3`

* Input args samples:

```
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
```

device_type: `llvm`(cpu) / `cuda` / `cuda-cudnn`

autotune: `True` / `False`

###### !!! The fastest method is cuda with autotuning acceleration while you have to run `python autotuning.py` first.

* Output samples: (bbox coordinates with confidences and label)


There is a sample code for you. `yolov3_example.py`

* Autotvm usage: `python autotuning.py` to generate the log file.

###### !!! It takes times.

## Results:

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |

    
