# Identification of Driver Phone Usage Violations using YOLOv3 and DeepSort

YOLOv3 implementation in TensorFlow 2.2.0, with training and object tracking

## Download YOLOv3 weights
wget -P model_data https://pjreddie.com/media/files/yolov3.weights

File structure should be arranged like this:
```bash
└───Directory
    ├───checkpoints
    ├───deep_sort
    │   └───__pycache__
    ├───IMAGES
    │   ├───detections
    │   │   └───summary
    │   └───to_train
    │       └───Archive
    ├───log
    ├───mAP-master
    │   ├───input
    │   │   ├───detection-results
    │   │   │   └───archive
    │   │   ├───ground-truth
    │   │   │   ├───archive
    │   │   │   └───backup
    │   │   └───images-optional
    │   │       └───archive
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
```

## Prepare images
For Google Open Images - see OIDv4 ToolKit
'old_to_pascal.py' to convert old Pascal to XML 
For bespoke images - Manually annotate using 'Labelimg'
Run 'XML_to_YOLOv3' to convert XML to Yolov3 format

Once XML files have been created run 'XML_to_YOLOv3' in tools directory

## Train model
Set ANNOT paths in config.py in YOLOv3 directory
Train model using 'train.py'
tensorboard --logdir=log
Track training progress in Tensorboard and go to http://localhost:6006/:

## AP and mAP
Test images saved in mAP-master/input/images-optional
Annotations (Pascal format) saved in mAP-master/input/ground-truth (file names to be same name as image file)
Run 'get_detection_results.py' to create detections files
Set CWD to ./mAP-master and run 'main.py'

## Run model with object tracking
Run 'object_tracker.py'

## References
Cloned and modified from https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3 
For downloading stock images from Google Image Dataset - https://github.com/EscVM/OIDv4_ToolKit.git
For mAP - https://github.com/Cartucho/mAP