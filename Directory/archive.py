# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 18:39:59 2021

@author: Steve
"""


# =============================================================================
# Function to calculate IOU scores from ground truth and predicted boxes. Function will also return any 
# false nagative and false positve. Note that this will only work on test 
# data with a max of 1 detection for each of the 2 classes. 
# =============================================================================
def get_iou_scores(gt_boxes, predicted_boxes, iou_thresh = 0.5):
    classes = ['Phone', 'Plate']  #Save classes so we can call by index
    dic = {'Phone':[], 'Plate':[]} #Creates dictionary to store results
    dic2 = {'Phone':[], 'Plate':[]}
    
    for i in range(0, len(gt_boxes)):
# =============================================================================
#       First condition checks the gt and predicted boxes where we have a perfect match. Will then pull
#       out any of legth 1 with different classes. This will return 0 for the IOU. The remainder (else)  
#       will then calculate the IOU for each of the classes.
# =============================================================================
        
        if len(gt_boxes[i]) == len(predicted_boxes[i]):
            #If below true, then the result is a false negative
            if len(gt_boxes[i]) == 1 and len(predicted_boxes[i]) == 1 and get_class(gt_boxes[i]) != get_class(predicted_boxes[i]):
                dic[classes[gt_boxes[i][0][-1]]].append([i+1, predicted_boxes[i][0][4], 'FN'])
                           
            else:    
                for box in gt_boxes[i]:
                    if box[-1]==0:
                        for detection in predicted_boxes[i]:
                            if detection[-1]==0:
                                 dic[classes[0]].append([i+1, detection[4], bb_intersection_over_union(detection[:4], box[:4])])
                    elif box[-1]==1:
                        for detection in predicted_boxes[i]:
                            if detection[-1]==1:
                                 dic[classes[1]].append([i+1, detection[4], bb_intersection_over_union(detection[:4], box[:4])])
                    else:
                        print('YES', i)
# =============================================================================
#       Next 2 conditions check for any missed predictions and then give return as false negative if so   
# =============================================================================
        elif get_class(predicted_boxes[i]).count(0) < 1 and get_class(gt_boxes[i]).count(0) > 0:
            if len(predicted_boxes[i]) > 0:
                for detection in predicted_boxes[i]:
                    for box in gt_boxes[i]:
                        if detection[-1] == box[-1] and detection[-1] ==1:
                             dic[classes[1]].append([i+1, detection[4], bb_intersection_over_union(detection[:4], box[:4])])                       
                        else:    
                            print(f'GOT A ZERO PHONE on {i}!!')
                            
                            dic[classes[0]].append([i+1, detection[4], 'FN'])
            else:
                print(predicted_boxes[i], i)
                dic[classes[gt_boxes[i][0][-1]]].append([i+1,  0, 'FN'])
                print(f'GOT A ZERO PHONE on {i}!!')
        elif get_class(predicted_boxes[i]).count(1) < 1 and get_class(gt_boxes[i]).count(1) > 0:
            if len(predicted_boxes[i]) > 0:
                for detection in predicted_boxes[i]:
                    for box in gt_boxes[i]:
                        if detection[-1] == box[-1] and detection[-1] ==0:
                             dic[classes[0]].append([i+1, detection[4], bb_intersection_over_union(detection[:4], box[:4])])                       
                        else:    
                            print(f'GOT A ZERO PLATE on {i}!!')
                            dic[classes[1]].append([i+1,  detection[4], 'FN'])
            else:
                dic[classes[gt_boxes[i][0][-1]]].append([i+1, 0, 'FN'])
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
                                dic[classes[box[-1]]].append([i+1, duplicate[4], bb_intersection_over_union(duplicate[:4], box[:4])])
                                print('HERE IT IS', i)
                dic[classes[0]].append([i+1, predicted_boxes[i][0][4], max(temp_iou_scores)])           
                print(f'DUPLICAT PHONE {i}')
            elif get_class(predicted_boxes[i]).count(1) > get_class(gt_boxes[i]).count(1) and get_class(predicted_boxes[i]).count(1) > 1:
                temp_iou_scores = []
                for duplicate in predicted_boxes[i]:
                    if duplicate[-1] == 1:
                        for detection in gt_boxes[i]:
                            if detection[-1] == 1:
                                temp_iou_scores.append(bb_intersection_over_union(detection[:4], duplicate[:4]))
                dic[classes[1]].append([i+1, predicted_boxes[i][0][4], max(temp_iou_scores)]) 
                print(f'DUPLICAT PLATE {i}')
            else:
                
                for detection in predicted_boxes[i]:
                    if detection[-1] == gt_boxes[i][0][-1]:
                        dic[classes[int(detection[-1])]].append([i+1, detection[4], bb_intersection_over_union(detection[:4], gt_boxes[i][0][:4])])
                        #print(f'Indes{i}', gt_boxes[i])
                        #print(f'Indes{i}', detection)
                    else:
                        dic[classes[int(detection[-1])]].append([i+1, detection[4], 'FP'])
            
    
    return dic   
            


results = get_iou_scores(gt_data, pred_data)