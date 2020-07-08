import cv2
path = '/Users/carrell/Desktop/image_det_2/TensorFlow-2.x-YOLOv3/IMAGES/seat_belt2/car2.jpg'

img = cv2.imread(path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('/Users/carrell/Desktop/image_det_2/TensorFlow-2.x-YOLOv3/IMAGES/img-png-color.jpg', 
            img_gray)  
