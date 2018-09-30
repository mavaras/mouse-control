# CALIBRATION MODULE

import cv2
import numpy as np
from skimage.filters.rank import maximum, minimum, median
from skimage.morphology import square
from skimage.exposure import equalize_hist
from skimage.color import rgb2gray


def nothing(x):
    pass;

#color = np.zeros((300, 512, 3), np.uint8);
kernel = np.ones((5,5),np.uint8);

name = "calibration";
cv2.namedWindow(name); cv2.resizeWindow(name, 400, 400);

cv2.createTrackbar("R", name, 0, 255, nothing);
cv2.createTrackbar("G", name, 0, 255, nothing);
cv2.createTrackbar("B", name, 0, 255, nothing);
#cv2.createTrackbar("0 : OFF \n1 : ON", name, 0, 1, nothing);

cap = cv2.VideoCapture(0);
while(1):
    ret, frame = cap.read();
    frame = cv2.flip(frame, 1);
    gray_frame = rgb2gray(frame);
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV);
    hsv = frame;
    # get trackbars vaules
    R = cv2.getTrackbarPos("R", name);
    G = cv2.getTrackbarPos("G", name);
    B = cv2.getTrackbarPos("B", name);
    #color[:] = [hue, sat, value];
    
    lower = np.array([R,G,B-100]);
    upper = np.array([R,G,B+100]);
    
    mask = cv2.inRange(hsv, lower, upper);
    #mask = cv2.equalizeHist(mask);
    """
    # Opening
    mask_eroded = cv2.erode(mask, np.ones((7,7),np.uint8), iterations=1);
    mask_dilated = cv2.dilate(mask_eroded, np.ones((5,5),np.uint8), iterations=1);
    
    # Closing
    mask_dilated = cv2.dilate(mask_dilated, np.ones((7,7),np.uint8), iterations=1);
    mask_eroded = cv2.erode(mask_dilated, np.ones((5,5),np.uint8), iterations=1);
    
    mask = mask_dilated;
    """
    #_,contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);
    #cv2.drawContours(frame, contours, -1, (0,255,0), 2);

    cv2.circle(hsv, (50,50), 35, [R-20, G, B], -1);
    #cv2.circle(frame, (50,50), 10, [hue, sat, value], -1);
    cv2.circle(hsv, (150,50), 35, [R+20, G, B], -1);

    cv2.imshow(name, mask);
    cv2.imshow("frame", hsv);
    #cv2.imshow("color", color);

    #res = cv2.bitwise_and(frame, frame, mask = mask);

    #cv2.imshow("frame", res);
    #cv2.imshow("frame", frame);
    #cv2.imshow("res", res);
    
    # ESC for exit
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows();