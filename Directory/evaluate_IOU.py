# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 18:21:47 2021

@author: Steve
"""

import os
import cv2
from yolov3.utils import load_yolo_weights, image_preprocess, postprocess_boxes, nms
from yolov3.yolov3 import Create_Yolov3
from yolov3.configs import *
import numpy as np
import tensorflow as tf


input_size = YOLO_INPUT_SIZE

yolo = Create_Yolov3(input_size=input_size, CLASSES=TRAIN_CLASSES)
yolo.load_weights("./checkpoints/yolov3_custom_Phone_Plate") 

test_annot_path = "model_data/dataset_test.txt"
#img_path = 'C:\\Users\\Steve\\Desktop\\deeplearning\\directory\\OIDv4_ToolKit-master\\OID\\Dataset/test\\Phone/mar1.JPG'

# =============================================================================
# Function to extract image path name and ground truth bounding box locations from the images
# =============================================================================
def extract_annot_data(test_annot_path):
    final_annotations = []
    with open(test_annot_path, 'r') as f:
        txt = f.readlines()
        annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        
    for annotation in annotations:
        # fully parse annotations
        line = annotation.split()
        #print(line[-2:])
        image_path, index = "", 1
        for i, one_line in enumerate(line):
            if not one_line.replace(",","").isnumeric():
                if image_path != "": image_path += " "
                image_path += one_line
            else:
                index = i
                break
        final_annotations.append([image_path, line[index:]])    
    
    return final_annotations


annotations = extract_annot_data(test_annot_path)


# =============================================================================
# Extracts and fomats the ground truth bb locations in useable format
# =============================================================================
def gt_bb_locations(annotations):
    gt_locations = []
    
    #Runs this block if we are dealing with more than 1 image
    if len(annotations) > 2:
        for i in range(0,len(annotations)):
            temp = annotations[i][1] #seperates only the bb locations for both classes
            image_location = []
            for box in temp:
                temp = box.split(',') #Splits each coordatate
                temp = [int(val) for val in temp] #Conversts from string to integer 
                image_location.append(np.array(temp)) #Appends as array in order to run calculations later
            gt_locations.append(image_location)  
    
    #Runs this block if only single image is given    
    else:    
        for box in annotations[1]:
            box = box.split(',')
            box = [int(val) for val in box]
            gt_locations.append(np.array(box))
    
    return gt_locations


# =============================================================================
# Gets predicted bounding box locations. Taken from detect image function and modified
# =============================================================================
def pred_bb_location(YoloV3, image_path, input_size=416, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.4, iou_threshold=0.45):
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
    
    locations = [np.delete(x, 4) for x in pred_locations] #Saves the predicted locations and class. Removes score threshold 
       
    return locations



def pred_bb_location_annotations(annotations):
    predicted_locations = []
    for i in range(0,len(annotations)):
        im_path = annotations[i][0]
        pred_location = pred_bb_location(yolo, im_path, input_size=input_size)
        predicted_locations.append(pred_location)
    return predicted_locations
    

gt_data = gt_bb_locations(annotations)
pred_data = pred_bb_location_annotations(annotations)

pred_data
# =============================================================================
# bb_intersection_over_union function copied from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/    
# =============================================================================
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou



#Simple function to count the classes. Used in the get_iou_scores function
def get_class(boxes):
    detection = []
    for det in boxes:
        detection.append(det[-1])
    return detection


# =============================================================================
# Function to calculate IOU scores from ground truth and predicted boxes. Note that this will only work on test 
# data with a max of 1 detection for each of the 2 classes
# =============================================================================
def get_iou_scores(gt_boxes, predicted_boxes):
    classes = ['Phone', 'Plate']  #Save classes so we can call by index
    dic = {'Phone':[], 'Plate':[]} #Creates dictionary to store results
    
    for i in range(0, len(gt_boxes)):
        if i == 17:
            print('INDEX 17 TIME!')
# =============================================================================
#       First condition checks the gt and predicted boxes where we have a perfect match. Will then pull
#       out any of legth 1 with different classes. This will return 0 for the IOU. The remainder (else)  
#       will then calculate the IOU for each of the classes.
# =============================================================================
        if len(gt_boxes[i]) == len(predicted_boxes[i]):
            if len(gt_boxes[i]) == 1 and len(predicted_boxes[i]) == 1 and get_class(gt_boxes[i]) != get_class(predicted_boxes[i]):
                dic[classes[gt_boxes[i][0][-1]]].append([0, i])
            else:    
                for box in gt_boxes[i]:
                    if box[-1]==0:
                        for detection in predicted_boxes[i]:
                            if detection[-1]==0:
                                 dic[classes[0]].append([bb_intersection_over_union(detection[:4], box[:4]), i])
                    elif box[-1]==1:
                        for detection in predicted_boxes[i]:
                            if detection[-1]==1:
                                 dic[classes[1]].append([bb_intersection_over_union(detection[:4], box[:4]), i])
                    else:
                        print('YES', i)
# =============================================================================
#         Next 2 conditions check for any missed predictions and then give those a score of zero    
# =============================================================================
        elif get_class(predicted_boxes[i]).count(0) < 1 and get_class(gt_boxes[i]).count(0) > 0:
            if len(predicted_boxes[i]) > 0:
                for detection in predicted_boxes[i]:
                    for box in gt_boxes[i]:
                        if detection[-1] == box[-1] and detection[-1] ==1:
                             dic[classes[1]].append([bb_intersection_over_union(detection[:4], box[:4]), i])                       
                        else:    
                            print(f'GOT A ZERO PHONE on {i}!!')
                            dic[classes[0]].append([0, i])
            else:
                dic[classes[gt_boxes[i][0][-1]]].append([0, i])
                print(f'GOT A ZERO PHONE on {i}!!')
        elif get_class(predicted_boxes[i]).count(1) < 1 and get_class(gt_boxes[i]).count(1) > 0:
            if len(predicted_boxes[i]) > 0:
                for detection in predicted_boxes[i]:
                    for box in gt_boxes[i]:
                        if detection[-1] == box[-1] and detection[-1] ==0:
                             dic[classes[0]].append([bb_intersection_over_union(detection[:4], box[:4]), i])                       
                        else:    
                            print(f'GOT A ZERO PLATE on {i}!!')
                            dic[classes[1]].append([0, i])
            else:
                dic[classes[gt_boxes[i][0][-1]]].append([0, i])
                print(f'GOT A ZERO PLATE on {i}!!')

# =============================================================================
#         Handles any where length of predicted are greater than ground truth. Also checks
#         for any duplicates and then calculates IOU on both and then takes the max. Else 
#         statement takes care of any where there is a predicted class without any corresponding
#         ground truth in the same class.
# =============================================================================
        elif len(predicted_boxes[i]) > len(gt_boxes[i]):

            if get_class(predicted_boxes[i]).count(0) > get_class(gt_boxes[i]).count(0) and get_class(predicted_boxes[i]).count(0) > 1:
                temp_iou_scores = []
                for duplicate in predicted_boxes[i]:
                    if duplicate[-1] == 0:
                        for detection in gt_boxes[i]:
                            if detection[-1] == 0:
                                temp_iou_scores.append(bb_intersection_over_union(detection[:4], duplicate[:4]))
                    else:
                        for box in gt_boxes[i]:
                            if duplicate[-1] == box[-1]:
                                dic[classes[box[-1]]].append([bb_intersection_over_union(duplicate[:4], box[:4]), i])
                                print('HERE IT IS', i)
                dic[classes[0]].append([max(temp_iou_scores), i])           
                print(f'DUPLICAT PHONE {i}')
            elif get_class(predicted_boxes[i]).count(1) > get_class(gt_boxes[i]).count(1) and get_class(predicted_boxes[i]).count(1) > 1:
                temp_iou_scores = []
                for duplicate in predicted_boxes[i]:
                    if duplicate[-1] == 1:
                        for detection in gt_boxes[i]:
                            if detection[-1] == 1:
                                temp_iou_scores.append(bb_intersection_over_union(detection[:4], duplicate[:4]))
                dic[classes[1]].append([max(temp_iou_scores), i]) 
                print(f'DUPLICAT PLATE {i}')
            else:
                for detection in predicted_boxes[i]:
                    if detection[-1] == gt_boxes[i][0][-1]:
                        dic[classes[gt_boxes[i][0][-1]]].append([bb_intersection_over_union(detection[:4], gt_boxes[i][0][:4]), i])

            
      
    return dic   
            


get_iou_scores(gt_data, pred_data)

