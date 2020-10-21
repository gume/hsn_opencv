
# coding: utf-8

# # Face Detection In Python Using OpenCV

import numpy as np
import cv2
import time
import sys


def detect_faces(f_cascade, colored_img, scaleFactor = 1.1):
    img_copy = np.copy(colored_img)
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    return img_copy

# XML training files for Haar cascade are stored in `opencv/data/haarcascades/` folder.

haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
lbp_face_cascade = cv2.CascadeClassifier('data/lbpcascade_frontalface.xml')

test = cv2.imread(sys.argv[1])
faces_detected_img = detect_faces(lbp_face_cascade, test)
cv2.imshow('Faces Image', faces_detected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

