import cv2
from google.colab.patches import cv2_imshow
import numpy as np

from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

img = cv2.imread('/content/car.jpg')

def draw(lh, ls, lv, hh, hs, hv):

    lower = np.array([lh, ls, lv], dtype = "uint8")
    higher = np.array([hh, hs, hv], dtype = "uint8")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]
    flt = cv2.inRange(hsv, lower, higher)

    cv2_imshow(img)
    #cv2_imshow(hsv)
    cv2_imshow(flt)


# interact(draw, lh=(0,255), ls=(0,255), lv=(0,255), hh=(0,255), hs=(0,255), hv=(0,255))

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Create mask for yellow
mask_yellow = cv2.inRange(hsv, (20, 0, 150),(60, 255, 255))
# Create mask for white
#mask_white = cv2.inRange(hsv, (0, 0, 160),(255, 9, 255))
# Union of the masks
mask = cv2.bitwise_or(mask_yellow, mask_white)
img2 = cv2.bitwise_and(img, img, mask=mask)

cv2_imshow(img2)

grayx = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# We need a gray image, but with full colors
gray = cv2.cvtColor(grayx, cv2.COLOR_GRAY2BGR)

cv2_imshow(gray)

# Create negative mask
nmask = cv2.bitwise_not(mask)

# Cretae the backgroung without the highlighted part
img3 = cv2.bitwise_and(gray, gray, mask=nmask)
cv2_imshow(img3)

# Create the highlighted part
img4 = cv2.bitwise_and(img, img, mask=mask)
cv2_imshow(img4)

# Merge gray background with the highlighted part
img5 = cv2.bitwise_or(img4, img3)

cv2_imshow(img5)
