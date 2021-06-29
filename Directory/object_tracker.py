#================================================================
#Based on https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#============================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import Load_Yolo_model, image_preprocess, postprocess_boxes, nms, draw_bbox, read_class_names
from yolov3.configs import *
import time

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
from datetime import datetime, date 
from time import gmtime, strftime  #Steven - added for sceduling the daily writing of detection_time_list to .txt file
import pandas as pd

#SEE IF THIS FIXES THE CUDNN ISSUE
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

yolo = Load_Yolo_model()

video_path = "./IMAGES/street2.avi"
video_path2 = 'IMAGES/vid_to_train/street1.avi'
csv_path = './detections/summary/' #Steven - set path for master CSV

def Object_tracking(Yolo, video_path, output_path, input_size=416, show=False, CLASSES=TRAIN_CLASSES, score_threshold=0.4, iou_threshold=0.45, rectangle_colors='', Track_only = []):
    # Definition of the parameters
    max_cosine_distance = 0.7
    nn_budget = None
    
    #initialize deep sort object
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    ID_LIST = [] #Steven - used to output images based on condition
    detection_time_list = [] #Steven - used to store all the detection times 
    
    times, times_2 = [], []

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
    fps_list = []
    
    current_day = datetime.today().strftime('%d/%m/%Y')
    current_time = str(strftime("%H:%M", gmtime()))
    current_hour = str(strftime("%H%p", gmtime()))
    phone_det_counter = 0
    plate_det_counter = 0
    
    frame_counter = 0
    
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
        
        #t1 = time.time()
        #pred_bbox = Yolo.predict(image_data)
        t2 = time.time()
        
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_frame, input_size, score_threshold)
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
        image = draw_bbox(original_frame, tracked_bboxes, CLASSES=CLASSES, tracking=True)

        t3 = time.time()
        times.append(t2-t1)
        times_2.append(t3-t1)
        
        times = times[-20:]
        times_2 = times_2[-20:]

        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        fps2 = 1000 / (sum(times_2)/len(times_2)*1000)
        
        image = cv2.putText(image, "Time: {:.1f} FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        # draw original yolo detection
        #image = draw_bbox(image, bboxes, CLASSES=CLASSES, show_label=False, rectangle_colors=rectangle_colors, tracking=True)
        fps_list.append(fps)
        #print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
        
        #Create new directory each day to store images of detections
        new_day = datetime.today().strftime('%d/%m/%Y')
        format_today = new_day.replace('/', '')
        
        if os.path.exists(f'./detections/{format_today}'):
            pass
        else:
            os.mkdir(f'./detections/{format_today}')
        
        #Add date and time to the image
        now = datetime.now()
        time_str = now.strftime("%d/%m/%Y %H:%M:%S")
        image = cv2.putText(image, time_str, (1000, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        
        #Below loop to check if ID has been seen, if not take snapshot and save in directory. ID will then be appended to the list to 
        #ensure more snapshots of the same ID are not saved.
        
        
        new_hour = str(strftime("%H", gmtime()))
        for id_no in ID:
            if id_no not in ID_LIST:
                det = 'Detection ' + class_name + ' ' + str(id_no) + ' at ' + time_str
                print(det)               
                detection_time_list.append(f'{det} Image location: ./detections/{format_today}/Detection {class_name} {str(id_no)}.0.jpg')
                ID_LIST.append(id_no)
                print(class_name)
                frame_counter = 20
                if class_name == 'Phone':
                    cv2.imwrite(f'./detections/{format_today}/Detection {class_name} {str(id_no)}.0.jpg', image)
                    phone_det_counter+=1
                    print('Phone count ' + str(phone_det_counter))
                else:
                    plate_det_counter+=1
                    print('Vehicle count ' + str(plate_det_counter))
        
        #Takes a further n snap shots in order to ensure a good image is taken            
        if frame_counter>0 and class_name == 'Phone':
            cv2.imwrite(f'./detections/{format_today}/Detection {class_name} {str(id_no)}.{20-frame_counter}.jpg', image)
            frame_counter-=1
            
                
        ##TO DO!!! Sort out the double appended hours whenever you re-fun program
        
        
        if new_hour != current_hour:
            if os.path.exists(csv_path):
                day_df = pd.DataFrame({"Date   ":[current_day], "Hour":[current_hour[0:2]], 
                 "Phone Detections":[phone_det_counter], "Vehicle Detections":[plate_det_counter]}) 
                master_csv = pd.read_csv('./detections/summary/tester.csv')
            
                master_csv = master_csv.append(day_df)
                master_csv.to_csv('./detections/summary/tester.csv', encoding='utf-8', index=False)
            else:
                os.mkdir('./detections/summary')
                master_csv = pd.DataFrame({"Date   ":[current_day], "Hour":[current_hour[0:2]], 
                 "Phone Detections":[phone_det_counter], "Vehicle Detections":[plate_det_counter]}) 
                master_csv.to_csv('./detections/summary/tester.csv', encoding='utf-8', index=False)
            
            #phone_det_counter = 0
            #plate_det_counter = 0
            current_hour = new_hour
            
        else:
            if os.path.exists(csv_path):
                master_csv = pd.DataFrame({"Date   ":[current_day], "Hour":[current_hour[0:2]], 
                                           "Phone Detections":[phone_det_counter], "Vehicle Detections":[plate_det_counter]}) 
                master_csv.to_csv('./detections/summary/tester.csv', encoding='utf-8', index=False)
            else:
                os.mkdir('./detections/summary')
                master_csv = pd.DataFrame({"Date   ":[current_day], "Hour":[current_hour[0:2]], 
                 "Phone Detections":[phone_det_counter], "Vehicle Detections":[plate_det_counter]}) 
                master_csv.to_csv('./detections/summary/tester.csv', encoding='utf-8', index=False)


        

        #Below code will write and save .txt file every 24 hours to show the daily detections which includes class, ID, time and also output 
        #file name. Daily detections added to the master CSV for futher analysis.
        new_time = str(strftime("%H:%M", gmtime()))
        
        
        #Changed to write each minute for the purpose of testing code. Need to remove 'str(current_time) and also change to
        #'if new_date != current date'
        #if new_day != current_day
        if new_time != current_time:
            no_detections = len(detection_time_list)
            detection_time_list.append('NUMBER OF DETECTIONS TODAY: ' + str(no_detections))

            with open(f"./detections/{format_today}_Hour_{str(current_time[0:2])}.txt", "w") as output:
                for row in detection_time_list:
                    output.write(str(row) + '\n')
            print('Saved new')
            detection_time_list = []
            current_time = new_time

        #****
        
        
        if output_path != '': out.write(image)
        if show:
            cv2.imshow('output', image)
            
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
         
    try:
        print(f'Average FPS is: {sum(fps_list)/len(fps_list)}')
    except:
        print('No video')        
    cv2.destroyAllWindows()



Object_tracking(yolo, video_path, "detection.mp4", input_size=YOLO_INPUT_SIZE, show=True, iou_threshold=0.1, rectangle_colors=(255,0,0), Track_only = [])


