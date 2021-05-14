# TensorFlow-2.x-YOLOv3 tutorial

YOLOv3 implementation in TensorFlow 2.2.0, with support for training, transfer training.

## Download YOLOv3 weights
wget -P model_data https://pjreddie.com/media/files/yolov3.weights

File structure should be arranged like this:
'''bash
─Directory
    ├───checkpoints
    ├───deep_sort
    │   └───__pycache__
    ├───IMAGES
    │   ├───detections
    │   │   └───summary
    │   └───to_train
    │       └───Archive
    ├───log
    ├───model_data
    │   └───coco
    ├───OIDv4_ToolKit-master
    │   ├───images
    │   ├───modules
    │   │   └───__pycache__
    │   └───OID
    │       ├───csv_folder
    │       └───Dataset
    │           ├───test
    │           │   ├───archive
    │           │   │   ├───Phone
    │           │   │   └───Vehicle_registration_plate
    │           │   ├───Phone
    │           │   └───Vehicle_registration_plate
    │           └───train
    │               ├───Phone
    │               │   └───Label
    │               └───Vehicle_registration_plate
    │                   └───Label
    ├───tools
    └───yolov3
        └───__pycache__
'''

python train.py
tensorboard --logdir=log
```
Track training progress in Tensorboard and go to http://localhost:6006/:
<p align="center">
    <img width="100%" src="IMAGES/tensorboard.png" style="max-width:100%;"></a>
</p>


## Custom Yolo v3 object detection training
Custom training required to prepare dataset first, how to prepare dataset and train custom model you can read in following link:<br>
https://pylessons.com/YOLOv3-TF2-custrom-train/

## Google Colab Custom Yolo v3 training
To learn more about Google Colab Free gpu training, visit my [text version tutorial](https://pylessons.com/YOLOv3-TF2-GoogleColab/)

## Yolo v3 Tiny train and detection
To get detailed instructions how to use Yolov3-Tiny, follow my text version tutorial [YOLOv3-Tiny support](https://pylessons.com/YOLOv3-TF2-Tiny/). Short instructions:
- Get YOLOv3-Tiny weights: ```wget -P model_data https://pjreddie.com/media/files/yolov3-tiny.weights```
- From `yolov3/configs.py` change `TRAIN_YOLO_TINY` from `False` to `True`
- Run `detection_demo.py` script.

## Yolo v3 Object tracking
To learn more about Object tracking with Deep SORT, visit [Following link](https://pylessons.com/YOLOv3-TF2-DeepSort/).
Quick test:
- Clone this repository;
- Make sure object detection works for you;
- Run object_tracking.py script
<p align="center">
    <img src="IMAGES/tracking_results.gif"></a>
</p>

