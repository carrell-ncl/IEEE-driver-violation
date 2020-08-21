# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 09:22:12 2020

@author: Steve
"""

import cv2
import os

vid = cv2.VideoCapture('./IMAGES/lightson2.asf')
im = cv2.imread('./IMAGES/yoda.jpg')
cv2.imshow('img',im)

while True:
    ret, frame = vid.read()
    if ret == True:
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        

cv2.waitKey(0)
cv2.destroyAllWindows()


with open("./IMAGES/detections/tester2.txt", "w") as output:
    output.write('HI2')
    
pm = ['13','14','15','16']
am = ['9','10','11','12']

num = '13:30'

num = str(num[0:2])

if num in pm:
    print('PM')
else:
    print('AM')
    
from time import gmtime, strftime
from datetime import datetime, date 
today = date.today()
print(today)
datetime.strptime(str(today), "%Y/%m/%d").strftime("%d-%m-%Y")
today = datetime.today().strftime('%d/%m/%Y')
today
day = today.strftime('%x')
day
current_time = str(strftime("%H:%M%p", gmtime()))
current_time[5:7]
type(day)
d = '15/08/2020'
d==today
d.replace('/', '')

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
