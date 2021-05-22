# Identification of Driver Phone Usage Violations using YOLOv3 and DeepSort

YOLOv3 implementation in TensorFlow 2.2.0, with training and object tracking

## Download YOLOv3 weights
wget -P model_data https://pjreddie.com/media/files/yolov3.weights

File structure should be arranged like this:
```bash
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


## Run model with object tracking
Run 'object_tracker.py'

## References
Cloned and modified from https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3 
For downloading stock images from Google Image Dataset - https://github.com/EscVM/OIDv4_ToolKit.git