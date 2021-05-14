# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 18:21:47 2021

@author: Steve
"""

import os
import cv2
from yolov3.utils import load_yolo_weights, image_preprocess, postprocess_boxes, nms, draw_bbox
from yolov3.yolov3 import Create_Yolov3
from yolov3.configs import *
import numpy as np
import tensorflow as tf
from scipy.integrate import trapz, simps
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapz

#Solves the CUDNN error issue
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

input_size = YOLO_INPUT_SIZE

yolo = Create_Yolov3(input_size=input_size, CLASSES=TRAIN_CLASSES)
yolo.load_weights("./checkpoints/yolov3_custom_Phone_Plate") 

#test_annot_path = "model_data/dataset_test.txt"
test_annot_path = ('./model_data/dataset_test.txt')

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
    
    #Run only if output giving 2 instead of 1 (temp fix)
    for x in gt_locations:
        if x[0][-1] ==2:
            x[0][-1]=1
    
    return gt_locations


# =============================================================================
# Gets predicted bounding box locations. Taken from detect image function and modified
# =============================================================================
def pred_bb_location(YoloV3, image_path, input_size=416, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45):
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
    
    #locations = [np.delete(x, 4) for x in pred_locations] #Saves the predicted locations and class. Removes score threshold 
       
    return pred_locations



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



def get_detection_results(gt_data, pred_data, class_ID=0):
    results = []
    
    for i in range(len(gt_data)):
    # =============================================================================
    #     Below block of code deals with >1 detection of the same class. Tests to see if it is part of the same detection (IOU >0), if so
    #     it will then append the max of these scores. For any scores below IOU of 0, we class these as a false positive
    # =============================================================================
        if get_class(pred_data[i]).count(class_ID) > get_class(gt_data[i]).count(class_ID) and get_class(pred_data[i]).count(class_ID) > 1:
            temp_iou = []
            for detection in pred_data[i]:
                
                if detection[-1] == class_ID and gt_data[i][0][-1] ==class_ID:
                    iou = bb_intersection_over_union(gt_data[i][0][:4], detection[:4])
                    #If IOU score is zero, we know it is outside the area and is a false positive.
                    if iou == 0:
                        results.append([i+1, detection[4], 'FP'])
                    else:
                        temp_iou.append(iou)
                        #print(iou, i)
            #Appends the max of the IOU scores (that are above zero)
            try:        
                results.append([i+1, detection[4], max(temp_iou)])
            except:
                pass
    # =============================================================================
    #     Below handles images where the length of gt and predictions are the same then checks if their class is the same.
    #     if true, it will calculate IOU, if false it will return false negative
    # =============================================================================
        elif len(gt_data[i]) ==1 and len(pred_data[i]) == 1:
            if gt_data[i][0][-1]==class_ID and pred_data[i][0][-1] ==class_ID:
                iou = bb_intersection_over_union(gt_data[i][0][:4], pred_data[i][0][:4])
                results.append([i+1, pred_data[i][0][4], iou])
            #Below finds where no prediction was made for the selected class and returns false negative
            else:
                if gt_data[i][0][-1]==class_ID and not gt_data[i][0][-1] == pred_data[i][0][-1]:
                    results.append([i+1, pred_data[i][0][4], 'FN'])
    # =============================================================================
    #     Handles the remainder of the images and iterates through the detections to find the matching classes and then 
    #     calculates IOU for them.        
    # =============================================================================
        else:
            for detection in pred_data[i]:
                for gt_box in gt_data[i]:
                    if detection[-1] == class_ID and gt_box[-1] == class_ID:
                        iou = bb_intersection_over_union(gt_box[:4], detection[:4])
                        results.append([i+1, detection[4], iou])
    return results

            
results_phone = get_detection_results(gt_data, pred_data, 0)
results_plate = get_detection_results(gt_data, pred_data, 1)


        


# =============================================================================
# Below function to calculate Average precision. Arguments are class and IOU threshold
# =============================================================================
def calculate_AP(class_ID=0, IOU_threshold=0.4):
    results = get_detection_results(gt_data, pred_data, class_ID)
    detections = []
    for val in results:
        try:
            if val[-1] >= IOU_threshold:
                val[-1]='TP'
                detections.append(val)
            else:
                val[-1]='FP'
                detections.append(val)
        except:
            print('NO IOUs')
            detections.append(val)
    
    #Change to dataframe and create columns    
    df = pd.DataFrame(detections, columns=['Image', 'Score', 'Detection']) 
    df = df.sort_values(by='Score', ascending=False)
    
    #Creates extra columns for TP, FP and FN columns
    df['TP'] = df.apply(lambda x: 1 if x['Detection']=='TP' else 0, axis=1)
    df['FP'] = df.apply(lambda x: 1 if x['Detection']=='FP' else 0, axis=1)
    df['FN'] = df.apply(lambda x: 1 if x['Detection']=='FN' else 0, axis=1)
    # Calculate precision and recall and put results into new columns
    df['Precision'] = df['TP'].cumsum()/(df['TP'].cumsum() + df['FP'].cumsum())
    # =============================================================================
    # For recall we can simply use length of DF to get the total number of possible positive due to there being
    # a max of 1 GT class per image (as mentioned earlier)
    # =============================================================================
    df['Recall'] = df['TP'].cumsum()/len(df) 
    prec =  list(df.Precision)
    #Start X at zero to allow us to calculate area under the line.
    prec.insert(0, 1)  
    rec = list(df.Recall)
    rec.insert(0,0)
    
    #Uncomment below to display recall/precision plot
    plt.plot(rec, prec)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
       
    #Calculates area under the line (AR)
    return trapz(prec, rec)    
            


# =============================================================================
# Calculate mAP using the PASCAL VOC method of IoU threshold of 0.5
# =============================================================================
AP_phone = calculate_AP(0, IOU_threshold=0.4)
AP_plate = calculate_AP(1, IOU_threshold=0.6)

print(AP_phone)


print(f'mAP is : {(AP_phone+AP_plate)/2}')

len(pred_data)

# =============================================================================
# Uncomment below to calculate mAP using COCO challenge metric with IoU ranging from 0.5 to 0.
# =============================================================================

# =============================================================================
# iou_vals = [0.5, 0.55, 0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
# 
# AP_scores_phone = []
# AP_scores_plate = []
# for vals in iou_vals:
#     AP_scores_phone.append(calculate_AP(0, vals))
#     AP_scores_plate.append(calculate_AP(1, vals))
# 
# av_AP_phone = sum(AP_scores_phone)/len(AP_scores_phone)
# av_AP_plate = sum(AP_scores_plate)/len(AP_scores_plate)
# 
# print(f'mAP is : {(av_AP_phone+av_AP_plate)/2}')
# =============================================================================

# =============================================================================
# Uncomment to display test images with GT and predicted boxes
# =============================================================================
# =============================================================================
# def detect_image(YoloV3, image_path, output_path, cor1, cor2, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):
#     original_image      = cv2.imread(image_path)
#     original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
#     original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
# 
#     image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
#     image_data = tf.expand_dims(image_data, 0)
# 
#     pred_bbox = YoloV3.predict(image_data)
#     pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
#     pred_bbox = tf.concat(pred_bbox, axis=0)
#     
#     bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
#     bboxes = nms(bboxes, iou_threshold, method='nms')
# 
#     print(bboxes)
#     image = draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
#     
# 
#     # Draw a rectangle with blue line borders of thickness of 2 px
#     image = cv2.rectangle(image, cor1, cor2, color=(0,255,0))
#     
# 
#     if output_path != '': cv2.imwrite(output_path, image)
#     if show:
#         # Show the image
#         cv2.imshow("predicted image", image)
#         # Load and hold the image
#         cv2.waitKey(0)
#         # To close the window after the required kill value was provided
#         cv2.destroyAllWindows()
#         
#     return image
# 
# 
# img_path = 'C:\\Users\\Steve\\Desktop\\deeplearning\\directory\\OIDv4_ToolKit-master\\OID\\Dataset/test\\Phone/mar23.JPG'
# directory = 'C:\\Users\\Steve\\Desktop\\deeplearning\\directory\\OIDv4_ToolKit-master\\OID\\Dataset/test\\Phone'
# 
# cor1 = tuple(gt_data[38][0][:2])
# cor2 = tuple(gt_data[38][0][2:4])
# 
# detect_image(yolo, img_path, "./IMAGES/det1.jpg", cor1=cor1, cor2=cor2, input_size=input_size, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
# 
# count = 0
# for file, coors in zip(os.listdir(directory), gt_data):
#     if file.endswith(".JPG") or file.endswith(".JPEG"):
#         image_path = os.path.join(directory, file)
#         print(image_path)
#         cor1 = tuple(gt_data[count][0][:2])
#         cor2 = tuple(gt_data[count][0][2:4])
#         print(cor1, cor2, pred_data[count])
#         detect_image(yolo, image_path, "./IMAGES/det1.jpg", cor1=cor1, cor2=cor2, input_size=input_size, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
#         count+=1
# 
# =============================================================================

