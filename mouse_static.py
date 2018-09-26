import cv2
import numpy as np
from scipy.ndimage import imread
from matplotlib.pyplot import figure, imshow
import math


frame = imread("foto.jpg");

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV);

# define range of blue color in HSV (tono, saturación, valor)
lower = np.array([0,10,70]); # 90 200
upper = np.array([15,155,155]);

# Threshold the HSV image to get only red colors
mask = cv2.inRange(frame, lower, upper);

# Bitwise-AND mask and original image
res = cv2.bitwise_and(frame, frame, mask = mask);

imshow(mask, cmap = "gray"); figure();
imshow(res, cmap = "gray"); figure();
imshow(frame); figure();       