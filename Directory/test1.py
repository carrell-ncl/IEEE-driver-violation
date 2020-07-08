#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 22:39:00 2020

@author: carrell
"""


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
from yolov3.yolov3 import Create_Yolov3
from yolov3.utils import load_yolo_weights, detect_image, detect_video, detect_realtime
from yolov3.configs import *
import tensorflow as tf


input_size = YOLO_INPUT_SIZE
Darknet_weights = YOLO_DARKNET_WEIGHTS



yolo = Create_Yolov3(input_size=input_size)
load_yolo_weights(yolo, Darknet_weights) # use Darknet weights


print(tf.__version__)
