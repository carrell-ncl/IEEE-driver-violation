#================================================================
#
#   File name   : detection_demo.py
#   Author      : PyLessons
#   Created date: 2020-05-18
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : object detection image and video example
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.yolov3 import Create_Yolov3
from yolov3.utils import load_yolo_weights, detect_image, detect_video, detect_realtime
from yolov3.configs import *

input_size = YOLO_INPUT_SIZE
Darknet_weights = YOLO_DARKNET_WEIGHTS
if TRAIN_YOLO_TINY:
    Darknet_weights = YOLO_DARKNET_TINY_WEIGHTS


img_path = 'C:\\Users\\Steve\\Desktop\\deeplearning\\directory\\OIDv4_ToolKit-master\\OID\\Dataset/test\\Phone/mar1.JPG'
image_path   = "./IMAGES/test_phone1.jpg"
#video_path   = "./IMAGES/lightson2.asf"
video_path = 'C:/Users/Steve/Documents/Bandicut/avigilon.avi'

yolo = Create_Yolov3(input_size=input_size, CLASSES=TRAIN_CLASSES)
yolo.load_weights("./checkpoints/yolov3_custom_Phone_Plate") # use keras weights

detect_image(yolo, img_path, "./IMAGES/det1.jpg", input_size=input_size, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
#detect_video(yolo, video_path, './IMAGES/detected.mp4', input_size=input_size, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
#detect_realtime(yolo, '', input_size=input_size, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))


# =============================================================================
# gpus = tf.config.experimental.list_physical_devices('GPU')
# gpus
# len(gpus)
# 
# print('GPU is AVAILABLE' if tf.test.is_gpu_available() else 'NOT AVAILABLE')
# 
# 
# print(tf.test.is_built_with_cuda()) 
# print(tf.config.list_physical_devices('GPU'))
# =============================================================================


