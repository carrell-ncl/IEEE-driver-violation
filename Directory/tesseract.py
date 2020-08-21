# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 20:42:59 2020

@author: Steve
"""

import tensorflow as tf
import cv2
import pytesseract as pt

path = 'C:/Program Files/Tesseract-OCR/Tesseract.exe'
pt.pytesseract.tesseract_cmd = path

x1,y1,x2,y2 = 597, 542, 1017, 672
path = './IMAGES/plate8.jpg'
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
crop_img = img[y1:y2, x1:x2]
#cv2.imshow("raw", img)


ret, crop_img=cv2.threshold(crop_img, 100,255, cv2.THRESH_BINARY)

cv2.imshow("cropped", crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


