
# coding: utf-8

# Import libraries
import os
import cv2
import numpy as np
import sys

# Define paths
base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir + 'data/deploy.prototxt')
caffemodel_path = os.path.join(base_dir + 'data/weights.caffemodel')

# Read the model
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

image = cv2.imread(sys.argv[1])

(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

model.setInput(blob)
detections = model.forward()

# Create frame around face
for i in range(0, detections.shape[2]):
	box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
	(startX, startY, endX, endY) = box.astype("int")

	confidence = detections[0, 0, i, 2]

	# If confidence > 0.5, show box around face
	if (confidence > 0.5):
		cv2.rectangle(image, (startX, startY), (endX, endY), (255, 255, 255), 2)

cv2.imshow('Faces Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
