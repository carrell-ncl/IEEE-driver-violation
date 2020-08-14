#================================================================
#
#   File name   : object_tracker.py
#   Author      : PyLessons
#   Created date: 2020-06-23
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : code to track detected object from video or webcam
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.yolov3 import Create_Yolov3
from yolov3.utils import load_yolo_weights, image_preprocess, postprocess_boxes, nms, draw_bbox, read_class_names#, detect_image, detect_video, detect_realtime
import time
from yolov3.configs import *


from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
from datetime import datetime, date 
from time import gmtime, strftime  #Steven - added for sceduling the daily writing of detection_time_list to .txt file
import os.path
import pandas as pd



input_size = YOLO_INPUT_SIZE
Darknet_weights = YOLO_DARKNET_WEIGHTS
if TRAIN_YOLO_TINY:
    Darknet_weights = YOLO_DARKNET_TINY_WEIGHTS

video_path   = "./IMAGES/test.mp4"
csv_path = './IMAGES/detections/summary/tester.csv' #Steven - set path for master CSV

#yolo = Create_Yolov3(input_size=input_size)
yolo = Create_Yolov3(input_size=input_size, CLASSES=TRAIN_CLASSES)
#load_yolo_weights(yolo, Darknet_weights) # use Darknet weights
yolo.load_weights("./checkpoints/yolov3_custom") # use keras weights



def Object_tracking(YoloV3, video_path, output_path, input_size=416, show=False, CLASSES=TRAIN_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', Track_only = []):
    # Definition of the parameters
    max_cosine_distance = 0.7
    nn_budget = None
    
    #initialize deep sort object
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    times = []
    ID_LIST = [] #Steven - used to output images based on condition
    detection_time_list = [] #Steven - used to store all the detection times 
    

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
    
    current_time = str(strftime("%H:%M", gmtime()))
    current_hour = str(strftime("%H", gmtime()))
    det_counter = 0
    
    while True:
        _, img = vid.read()

        try:
            original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        except:
            break
        image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
        image_data = tf.expand_dims(image_data, 0)
        
        t1 = time.time()
        pred_bbox = YoloV3.predict(image_data)

        t2 = time.time()


        times.append(t2-t1)
        times = times[-20:]
        
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
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
        features = np.array(encoder(original_image, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]
        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)
       

        # Obtain info from the tracks
        tracked_bboxes = []
        ID = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue 
            bbox = track.to_tlbr() # Get the corrected/predicted bounding box
            class_name = track.get_class() #Get the class name of particular object
            tracking_id = track.track_id # Get the ID for the particular track
            index = key_list[val_list.index(class_name)] # Get predicted object index by object name
            #Added scores to tracked boxes so we can see probability confidence of each detection
            tracked_bboxes.append(bbox.tolist() + [tracking_id, index, scores]) # Structure data, that we could use it with our draw_bbox function
            ID.append(tracking_id)

            
            
        
        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
    

        # draw detection on frame
        image = draw_bbox(original_image, tracked_bboxes, CLASSES=CLASSES, tracking=True)
        image = cv2.putText(image, "FPS: {:.1f}".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        
        #****Steven - add date and time to image
        today = date.today()
        now = datetime.now()
        time_str = now.strftime("%d/%m/%Y %H:%M:%S")
        image = cv2.putText(image, time_str, (1000, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        #Below loop to check if ID has been seen, if not take snapshot and save in directory. ID will then be appended to the list to 
        #ensure more snapshots of the same ID are not saved.
        
        new_hour = str(strftime("%H", gmtime()))
        for name, id_no in zip(names, ID):
            if id_no not in ID_LIST:
                det = 'Detection ' + name + ' ' + str(id_no) + ' at ' + time_str
                print(det)               
                cv2.imwrite('./IMAGES/detections/detection'+str(id_no)+'.jpg', image)
                detection_time_list.append(det + ' Image location: ' + './IMAGES/detections/detection'+str(id_no)+'.jpg')
                ID_LIST.append(id_no)
                det_counter+=1
                print(det_counter)
                
        if new_hour != current_hour:
            if os.path.exists(csv_path):
                day_df = pd.DataFrame({"Date":[str(today)], "Hour":[current_hour], 
                 "Detections":[det_counter]}) 
                master_csv = pd.read_csv('./IMAGES/detections/summary/tester.csv')
            
                master_csv = master_csv.append(day_df)
                master_csv.to_csv('./IMAGES/detections/summary/tester.csv', encoding='utf-8', index=False)
            else:
                master_csv = pd.DataFrame({"Date":[str(today)], "Hour":[current_hour], 
                 "Detections":[det_counter]}) 
                master_csv.to_csv('./IMAGES/detections/summary/tester.csv', encoding='utf-8', index=False)
            current_hour = new_hour
            

        #Below code will write and save .txt file every 24 hours to show the daily detections which includes class, ID, time and also output 
        #file name. Daily detections added to the master CSV for futher analysis.
        new_time = str(strftime("%H:%M", gmtime()))
        if new_time != current_time:
            
            no_detections = len(detection_time_list)
            detection_time_list.append('NUMBER OF DETECTIONS TODAY: ' + str(no_detections))
            with open("./IMAGES/detections/" + current_time +".txt", "w") as output:
                for row in detection_time_list:
                    output.write(str(row) + '\n')
            print('Saved new')
            detection_time_list = []
            current_time = new_time

        #****

        # draw original yolo detection
        #image = draw_bbox(image, bboxes, CLASSES=CLASSES, show_label=False, rectangle_colors=rectangle_colors, tracking=True)

        #print("Time: {:.2f}ms, {:.1f} FPS".format(ms, fps))q
        if output_path != '': out.write(image)
        if show:
            cv2.imshow('output', image)
            
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
    print(current_time)      
    cv2.destroyAllWindows()


#Object_tracking(yolo, video_path, "detection.mp4", input_size=input_size, show=True, iou_threshold=0.1, rectangle_colors=(255,0,0), Track_only = ["person"])
Object_tracking(yolo, video_path=False, output_path="detection_track_nofilt.mp4", input_size=input_size, show=True, iou_threshold=0.1, rectangle_colors=(255,0,0), Track_only = ['Phone'])


