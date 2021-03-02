# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 09:22:12 2020

@author: Steve
"""

import cv2
import imutils


vid = cv2.VideoCapture('./IMAGES/lightson2.asf')
vid2 = cv2.VideoCapture('rtsp://administrator:pass1@169.254.101.136/defaultPrimary?streamType=m')
x1,y1,x2,y2 = 597, 542, 1017, 672

#im = cv2.imread('./IMAGES/yoda.jpg')
#cv2.imshow('img',im)
count = 0
while True:
    ret, frame = vid2.read()
    if ret == True:
        frame = imutils.resize(frame, width=1600)
       # crop_img = frame[y1:y2, x1:x2]
        cv2.imshow('Frame', frame)
        #cv2.imshow('Frame2', crop_img)
        count +=1
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
        

        

cv2.waitKey(0)
cv2.destroyAllWindows()


