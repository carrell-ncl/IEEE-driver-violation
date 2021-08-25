#================================================================
#Based on https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#============================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image, detect_image2, detect_realtime, detect_video, detect_video2, Load_Yolo_model, detect_video_realtime_mp, load_yolo_weights, Load_Yolo_model2
from yolov3.configs import *
from yolov3.yolov4 import Create_Yolo

#Solves the CUDNN error issue
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

image_path   = "./IMAGES/jai2_Moment.jpg"
video_path   = "./IMAGES/drive6.mp4"
#video_path = './IMAGES/rotated2.mp4'
video_path2 = 'IMAGES/street2.avi'

coor = [350, 200, 700, 700]

#TRAIN_MODEL_NAME = 'yolov4_custom_PP2'
yolo = Load_Yolo_model()
yolo2 = Load_Yolo_model2()
detect_image2(yolo,yolo2, image_path, "./IMAGES/plate_1_detect.jpg", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
detect_video(yolo, video_path, './IMAGES/detected.mp4', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
#detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))

