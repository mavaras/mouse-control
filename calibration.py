# CALIBRATION MODULE

import cv2
import numpy as np


def nothing(x):
    pass;

#kernel = np.zeros((300, 512, 3), np.uint8);
kernel = np.ones((7,7),np.uint8);

name = "calibration";
cv2.namedWindow(name); cv2.resizeWindow(name, 400, 400);

cv2.createTrackbar("hue", name, 0, 180, nothing);
cv2.createTrackbar("sat", name, 0, 255, nothing);
cv2.createTrackbar("value", name, 0, 255, nothing);
#cv2.createTrackbar("0 : OFF \n1 : ON", name, 0, 1, nothing);

cap = cv2.VideoCapture(0);
while(1):
    ret, frame = cap.read();
    frame = cv2.flip(frame, 1);
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV);
     
    # get trackbars vaules
    hue = cv2.getTrackbarPos("hue", name);
    sat = cv2.getTrackbarPos("sat", name);
    value = cv2.getTrackbarPos("value", name);
    
    lower = np.array([hue-20,sat,value])
    upper = np.array([hue+20,255,255])
    
    mask = cv2.inRange(hsv, lower, upper);
    mask_eroded = cv2.erode(mask, kernel, iterations=1);
    mask_dilated = cv2.dilate(mask_eroded, kernel, iterations=1);
    mask = mask_dilated;

    cv2.imshow(name, mask);

    #res = cv2.bitwise_and(frame, frame, mask = mask);

    #cv2.imshow("frame", frame);
    #cv2.imshow("res", res);
    
    # ESC for exit
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows();