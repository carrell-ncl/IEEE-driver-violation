# Identification of Driver Phone Usage Violations using YOLOv3 and DeepSort

YOLOv4 implementation in TensorFlow 2.2.0, with training and object tracking
Training and testing done using GPU (Nvidia 2080ti)

![](capture3.gif)

## Download YOLOv3/v4 weights
# yolov3
wget -P model_data https://pjreddie.com/media/files/yolov3.weights

# yolov4
wget -P model_data https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

File structure should be arranged like this:
```bash
└───Directory
    ├───archived scripts
    ├───checkpoints
    ├───deep_sort
    │   └───__pycache__
    ├───detections

    ├───IMAGES
    │   ├───for the report
    │   ├───frames_to_test
    │   ├───vid_to_test
    │   └───vid_to_train
    ├───log
    ├───mAP
    │   └───ground-truth
    ├───mAP-master
    │   ├───input
    │   │   ├───detection-results
    │   │   │   └───archive
    │   │   ├───ground-truth
    │   │   │   ├───archive
    │   │   │   └───backup
    │   │   └───images-optional
    │   ├───output
    │   │   ├───classes
    │   │   └───images
    │   │       └───detections_one_by_one
    │   └───scripts
    │       └───extra
    ├───model_data
    │   └───coco
    ├───OIDv4_ToolKit-master
    │   ├───modules
    │   │   └───__pycache__
    │   └───OID
    │       ├───csv_folder
    │       └───Dataset
    │           ├───test
    │           │   ├───Phone
    │           │   └───Vehicle_registration_plate
    │           └───train
    │               ├───Phone
    │               │   └───Label
    │               └───Vehicle_registration_plate
    │                   └───Label
    ├───tools
    ├───yolov3
```

## Prepare images
For Google Open Images - see OIDv4 ToolKit \
'old_to_pascal.py' to convert old Pascal to XML \
For bespoke images - Manually annotate using 'Labelimg' \
Run 'XML_to_YOLOv3' to convert XML to Yolov3 format \

Once XML files have been created run 'XML_to_YOLOv3' in tools directory

![](annot.JPG)

## Train model
Set ANNOT paths in config.py in YOLOv3 directory \
Train model using 'train.py' \
tensorboard --logdir=log \
Track training progress in Tensorboard and go to http://localhost:6006\:

## AP and mAP
Test images saved in mAP-master/input/images-optional \
Annotations (Pascal format) saved in mAP-master/input/ground-truth (file names to be same name as image / file) \
Run 'get_detection_results.py' to create detections files \
Set CWD to ./mAP-master and run 'main.py'

## Run model with object tracking
Run 'object_tracker.py' \
** Need to talk about the unique detection capture and .csv log \

## References
Cloned and modified from https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3 \
For downloading stock images from Google Image Dataset - https://github.com/EscVM/OIDv4_ToolKit.git \
For mAP - https://github.com/Cartucho/mAP