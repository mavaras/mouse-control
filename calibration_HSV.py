# CALIBRATION MODULE

import cv2
import numpy as np
from skimage.filters import gaussian
from skimage.filters.rank import maximum, minimum, median, mean
from skimage.morphology import square
from skimage.exposure import equalize_hist
from skimage.color import rgb2gray


def nothing(x):
    pass;

#color = np.zeros((300, 512, 3), np.uint8);
kernel = np.ones((5,5),np.uint8);

name = "calibration";
cv2.namedWindow(name); cv2.resizeWindow(name, 400, 400);

cv2.createTrackbar("hue", name, 0, 180, nothing);
cv2.createTrackbar("sat", name, 0, 255, nothing);
cv2.createTrackbar("value", name, 0, 255, nothing);
#cv2.createTrackbar("0 : OFF \n1 : ON", name, 0, 1, nothing);

cap = cv2.VideoCapture(0);
while(1):
    ret, frame = cap.read();
    #frame = np.copy(frame);
    frame = np.array(frame, dtype = np.uint8);
    frame = cv2.flip(frame, 1);
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV);
    
    # get trackbars vaules
    hue = cv2.getTrackbarPos("hue", name);
    sat = cv2.getTrackbarPos("sat", name);
    value = cv2.getTrackbarPos("value", name);
    #color[:] = [hue, sat, value];
    
    lower = np.array([hue,sat,value]);
    upper = np.array([255,255,255]);
    
    mask = cv2.inRange(hsv, lower, upper);
    #mask = cv2.equalizeHist(mask);

    # Morphology: Opening + Close
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8));
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15,15),np.uint8));
    
    """
    # Opening
    #mask_eroded = cv2.erode(mask, np.ones((7,7),np.uint8), iterations=1);
    mask_dilated = cv2.dilate(mask, np.ones((5,5),np.uint8), iterations=1);
    
    # Closing
    mask_dilated = cv2.dilate(mask_dilated, np.ones((7,7),np.uint8), iterations=1);
    mask_eroded = cv2.erode(mask_dilated, np.ones((5,5),np.uint8), iterations=1);
    
    mask = mask_dilated;
    """

    # Working with contours of the mask
    _,contours,hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);
    cv2.drawContours(frame, contours, -1, (0,255,0), 2);
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c);
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2);
        #cv2.line(frame, (x, y), (x+w, y+h), (255, 0, 0), 2);
        #cv2.line(frame, (x + w, y), (x, y + h), (255, 0, 0), 2);

    cv2.circle(frame, (50,50), 35, [hue-20, sat, value], -1);
    #cv2.circle(frame, (150,50), 35, [hue+20, 255, 255], -1);

    cv2.imshow(name, mask);
    cv2.imshow("frame", frame);

    #res = cv2.bitwise_and(frame, frame, mask = mask);

    #cv2.imshow("frame", res);
    #cv2.imshow("frame", frame);
    #cv2.imshow("res", res);
    
    # ESC for exit
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break;

cap.release();
cv2.destroyAllWindows();