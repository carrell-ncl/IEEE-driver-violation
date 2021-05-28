# -*- coding: utf-8 -*-
"""
Created on Wed May 26 16:44:15 2021

@author: Steve
"""

import os
import cv2
from yolov3.utils import load_yolo_weights, image_preprocess, postprocess_boxes, nms, draw_bbox
from yolov3.yolov3 import Create_Yolov3
from yolov3.configs import *
from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#Solves the CUDNN error issue
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

input_size = YOLO_INPUT_SIZE

yolo = Create_Yolov3(input_size=input_size, CLASSES=TRAIN_CLASSES)
yolo.load_weights("./checkpoints/yolov3_custom_Phone_Plate2") 

image_path   = "mAP-master/input/images-optional/mar1.JPG"
image_directory = "mAP-master/input/images-optional"

dic = ['Phone', 'Vehicle_registration_plate']

# =============================================================================
# Gets predicted bounding box locations. Taken from detect image function and modified
# =============================================================================
def detection_results(YoloV3, image_path, image_name, input_size=416, CLASSES=TRAIN_CLASSES  , score_threshold=0.3, iou_threshold=0.45):
    original_image      = cv2.imread(image_path)
    original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = tf.expand_dims(image_data, 0)

    pred_bbox = YoloV3.predict(image_data)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    
    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
    
    pred_locations = nms(bboxes, iou_threshold, method='nms')
    
    #Save detection results in the required order/format
    myfile = open(f'mAP-master/input/detection-results/{image_name}.txt', 'w')
   
    for detection in pred_locations:
        output = list(detection[:5])
        output.insert(0, output.pop())
        #print(pred_locations[0][5])
        output.insert(0, dic[int(detection[5])])
    
        output = [str(x) for x in output]
        output = ' '.join(output)
        output +='\n'
        myfile.write(output)
        #print(output2)
    myfile.close()    



for filename in os.listdir(image_directory):
    if filename.endswith(".JPG") or filename.endswith(".JPEG") or filename.endswith(".jpg"):
        #detection = detection_results(yolo, filename)
        image_name = Path(os.path.join(image_directory, filename)).stem
        detection = detection_results(yolo, os.path.join(image_directory, filename), image_name)
print('All detection results now saved!')
        
        
        
def detect_image(YoloV3, image_path, output_path, input_size=416, show=False, CLASSES=TRAIN_CLASSES   , score_threshold=0.4, iou_threshold=0.45, rectangle_colors=''):
    original_image      = cv2.imread(image_path)
    original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = tf.expand_dims(image_data, 0)

    pred_bbox = YoloV3.predict(image_data)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    
    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='nms')


    image = draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)

    if output_path != '': cv2.imwrite(output_path, image)
    if show:
        # Show the image
        cv2.imshow("predicted image", image)
        # Load and hold the image
        cv2.waitKey(0)
        # To close the window after the required kill value was provided
        cv2.destroyAllWindows()
        
    return image

for filename in os.listdir(image_directory):
    if filename.endswith(".JPG") or filename.endswith(".JPEG") or filename.endswith(".jpg"):
        #detection = detection_results(yolo, filename)
        image_name = Path(os.path.join(image_directory, filename)).stem
        detection = detect_image(yolo, os.path.join(image_directory, filename), output_path='', show=True)

