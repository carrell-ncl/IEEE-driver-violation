# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 13:38:22 2021

@author: Steve
"""

#================================================================
#Based on https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#============================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import Load_Yolo_model,Load_Yolo_model2, image_preprocess, postprocess_boxes, nms, draw_bbox, read_class_names
from yolov3.configs import *
import time

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
from datetime import datetime, date 
from time import gmtime, strftime  #Steven - added for sceduling the daily writing of detection_time_list to .txt file
import pandas as pd

#FIXES THE CUDNN ISSUE
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

yolo = Load_Yolo_model() #Loads windscreen detector
yolo2 = Load_Yolo_model2() #Loads phone detector

video_path = "./IMAGES/drive13.mp4"
video_path2 = 'IMAGES/june_8_1.mp4'
csv_path = './detections/summary/tester.csv' #Set path for master CSV
            
def Object_tracking(Yolo, Yolo2, video_path, output_path, input_size=YOLO_INPUT_SIZE, input_size2=YOLO_INPUT_SIZE2, show=False, CLASSES=TRAIN_CLASSES, CLASSES2=TRAIN_CLASSES2, 
                    score_threshold_screen=0.3, score_threshold_phone=0.6, iou_threshold=0.45, Track_only = [], take_snapshots = True):
    # Definition of the parameters
    max_cosine_distance = 0.7
    nn_budget = None
    
    #initialize deep sort object
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

#Create variables to be used in function
    ID_list = [] #Steven - used to output images based on condition
    ID_list_vehicle_count = [] #- used to count vehicles
    vehicle_count = 0 #to log total number of vehicles
    phone_det_counter = 0
    temp_vehicle_counter = 0 #Keeps number of vehicles between detections
    counter = 1 #Index used for imnage snapshots
    frame_counter1 = 0 #Variable used to output violation ID text
    
# =============================================================================
#Times to be appended to work out average fps for det only, det + tracking, and det, tracking and two-step
# =============================================================================
    times, times2, times3 = [], [], [] 

    if video_path:
        vid = cv2.VideoCapture(video_path) # detect on video
    else:
        vid = cv2.VideoCapture(0) # detect from webcam

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4

    NUM_CLASS = read_class_names(CLASSES)
    key_list = list(NUM_CLASS.keys()) 
    val_list = list(NUM_CLASS.values())

    
    current_day = datetime.today().strftime('%d/%m/%Y')
    current_time = str(strftime("%H:%M", gmtime()))
    current_time_str = current_time.replace(':', '.')
    current_hour = str(strftime("%H%p", gmtime()))


    

    rectangle_colors = (255, 0, 0)
    while True:
        _, frame = vid.read()

        try:
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        except:
            break
        
        image_data = image_preprocess(np.copy(original_frame), [input_size, input_size])
        #image_data = tf.expand_dims(image_data, 0)
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        if YOLO_FRAMEWORK == "tf":
            pred_bbox = Yolo.predict(image_data)
        elif YOLO_FRAMEWORK == "trt":
            batched_input = tf.constant(image_data)
            result = Yolo(batched_input)
            pred_bbox = []
            for key, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)
        

        t2 = time.time()
        
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_frame, input_size, score_threshold_screen)
        bboxes = nms(bboxes, iou_threshold, method='nms')

        # extract bboxes to boxes (x, y, width, height), scores and names
        boxes, scores, names = [], [], []
        for bbox in bboxes:
            if len(Track_only) !=0 and NUM_CLASS[int(bbox[5])] in Track_only or len(Track_only) == 0:
                boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int)-bbox[0].astype(int), bbox[3].astype(int)-bbox[1].astype(int)])
                scores.append(bbox[4])
                names.append(NUM_CLASS[int(bbox[5])])

        # Obtain all the detections for the given frame.
        boxes = np.array(boxes) 
        names = np.array(names)
        scores = np.array(scores)
        features = np.array(encoder(original_frame, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]

        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)

        # Obtain info from the tracks
        tracked_bboxes = []
        ID =[]
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue 
            bbox = track.to_tlbr() # Get the corrected/predicted bounding box
            class_name = track.get_class() #Get the class name of particular object
            tracking_id = track.track_id # Get the ID for the particular track
            index = key_list[val_list.index(class_name)] # Get predicted object index by object name
            tracked_bboxes.append(bbox.tolist() + [tracking_id, index]) # Structure data, that we could use it with our draw_bbox function
            
            ID.append(tracking_id)
            
            
        # draw detection on frame
        image = draw_bbox(original_frame, tracked_bboxes, rectangle_colors=rectangle_colors, CLASSES=CLASSES, tracking=True)
        
        #Add date and time to the image
        now = datetime.now()
        time_str = now.strftime("%d/%m/%Y %H:%M:%S")
        image = cv2.putText(image, time_str, (800, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        
        t3 = time.time()
        times.append(t2-t1)
        times2.append(t3-t1)
        

       
        #Create new directory each day to store images of detections
        new_day = datetime.today().strftime('%d/%m/%Y')
        format_today = new_day.replace('/', '')
        
        if os.path.exists(f'./detections/{format_today}'):
            pass
        else:
            os.mkdir(f'./detections/{format_today}')
            
        if os.path.exists('./detections/summary'):
            pass
        else:
            os.mkdir('./detections/summary')

        for detection in tracked_bboxes:
            #print(detection[-2])
            try:
                
                obj_id = detection[4]
                #print(obj_id)
                
                
                output = list(detection[:4])
                output = [int(x) for x in output]
                cropped = original_frame[output[1]:output[3], output[0]:output[2]]
                cropped      = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                #cropped      = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                
                original_image2     = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                #original_image2     = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                image_data2 = image_preprocess(np.copy(original_image2), [input_size2, input_size2])
                image_data2 = image_data2[np.newaxis, ...].astype(np.float32)
                
                pred_bbox2 = Yolo2.predict(image_data2)
                pred_bbox2 = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox2]
                pred_bbox2 = tf.concat(pred_bbox2, axis=0)
                bboxes2 = postprocess_boxes(pred_bbox2, original_image2, input_size2, score_threshold_phone)
                bboxes2 = nms(bboxes2, iou_threshold, method='nms')
                image2 = draw_bbox(original_image2, bboxes2, CLASSES=CLASSES2, rectangle_colors=rectangle_colors)
                
                
                if obj_id not in ID_list_vehicle_count:

                    temp_vehicle_counter += 1
                    ID_list_vehicle_count.append(obj_id)
                
                #Draw phone bounding box on the original image
                for det in bboxes2:
                    output2 = list(det[:4])
                    output2 = [int(x) for x in output2]
        
                    new_coors = [output[0]+output2[0], output[1]+output2[1], (output2[2]-output2[0])+output[0]+output2[0], (output2[3]-output2[1])+output[1]+output2[1]]
                    image = cv2.rectangle(original_frame, tuple(new_coors[:2]), tuple(new_coors[2:]), (0,0,255), 1)
                
                    # Record each unique phone detection and add to CSV file
                    if obj_id not in ID_list:
                        ID_list.append(obj_id)
                        print('YES!!')
                        print('Phone violation!')
                        phone_det_counter+=1
                        #creates/updates CSV file with number of violations, number of vehicles and timestamps
                        if os.path.exists(csv_path):
    
                            day_df = pd.DataFrame({"Date   ":[current_day], "Time":[current_time], "Time Stamp":[time.time()],
                             "Phone Detections":[1], "Vehicle Detections":[temp_vehicle_counter]}) 
                            master_csv = pd.read_csv('./detections/summary/tester.csv')
                        
                            master_csv = master_csv.append(day_df)
                            master_csv.to_csv('./detections/summary/tester.csv', encoding='utf-8', index=False)
                        else:
                
                            #os.mkdir('./detections/summary/')
                            master_csv = pd.DataFrame({"Date   ":[current_day], "Time":[current_time], "Time Stamp":[time.time()],
                             "Phone Detections":[1], "Vehicle Detections":[temp_vehicle_counter]}) 
                            master_csv.to_csv('./detections/summary/tester.csv', encoding='utf-8', index=False)
                    


                    
                    #temp_vehicle_counter = 0 #Clears the vehicle count once phone violation detected
                     
                # Puts a red bounding box around windscreen where there is a violation
                if len(bboxes2) >0:
                    frame_counter1 = 10
                    image = cv2.rectangle(original_frame, tuple(output[:2]), tuple(output[2:]), (0,0,255), 3)
                    temp_vehicle_counter = 0 #Clears the vehicle count once phone violation detected
                    #outputs snapshots for violation. Saves into the daily folder
                    if take_snapshots == True:
                        
                        det = f'Vehicle ID {obj_id} at {current_time_str}hrs image {counter}'
                        #print(det)
                        counter+=1
                        cv2.imwrite(f'./detections/{format_today}/{det}.jpg', image)
                    
                ID_list = list(set(ID_list)) #Change to unique value for obj_id
            except:
                break
      
            vehicle_count = len(list(set(ID_list_vehicle_count)))
            #print(ID_list)
                                                                                                                                                                                                                                                                
            #Displays phone violation for n number of frames to avoid flashing
            if frame_counter1>0:
                image = cv2.putText(image, f'PHONE VIOLATION VEHICLE ID {obj_id}!', (1400, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
                frame_counter1-=1


                
        t4 = time.time()
        times3.append(t4-t1)

 #       times = times[-20:]      #Averages out the last 20 frames so it doesn't keep changing to frequent
  #      times2 = times2[-20:]
        

        fps = 1/(sum(times)/len(times)) #Calculate average FPS for detection only
        fps2 = 1/(sum(times2)/len(times2)) #Calculate for detection + tracking
       
        fps_full = 1/(sum(times3)/len(times3)) #Calculate for detection + tracking + two step
        
        image = cv2.putText(image, "Det only FPS: {:.2f}".format( 1/(sum(times[-20:])/len(times[-20:]))), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        image = cv2.putText(image, "FPS: {:.2f}".format(1/(sum(times3[-20:])/len(times3[-20:]))), (400, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        
        
        
        #Add date and time to the image
        now = datetime.now()
        time_str = now.strftime("%d/%m/%Y %H:%M:%S")
        image = cv2.putText(image, time_str, (800, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        
       
        if output_path != '': out.write(image)
        if show:
            cv2.imshow('output', image)
            
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
         
    try:
        #outputs the average FPS for detection only, detection and tracking, detection, tracking and 2 step
        print(f'Average FPS for detection only is: {fps}')
        print(f'Average FPS for detection and tracking is: {fps2}')
        print(f'Average FPS for detection, tracking and two step is: {fps_full}')
    except:
        print('No video')        
    cv2.destroyAllWindows()

    print(f'Total number of phone violations: {phone_det_counter}')
    print(f'Total number of vehicles: {vehicle_count}')


Object_tracking(yolo, yolo2,  video_path, "detection2.mp4", show=True, iou_threshold=0.1,  Track_only = [], take_snapshots=True)



l=[1,2,3,4,5,6,7,8,9]
l[-3:]
