# CALIBRATION MODULE

import cv2
import numpy as np


def nothing(x):
    pass;

kernel = np.zeros((300, 512, 3), np.uint8);

def calibrate_color(color, res_range):
    name = "calibration";
    cv2.namedWindow(name); cv2.resizeWindow(name, 400, 400);
    
    cv2.createTrackbar("hue", name, res_range[0][0], 180, nothing);
    cv2.createTrackbar("sat", name, res_range[0][1], 255, nothing);
    cv2.createTrackbar("value", name, res_range[0][2], 255, nothing);
    #cv2.createTrackbar("0 : OFF \n1 : ON", name, 0, 1, nothing);
    
    while(1):
        ret, frame = cap.read();
        frame = cv2.flip(frame, 1);
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV);
         
        # get trackbars vaules
        hue = cv2.getTrackbarPos("hue", name);
        sat = cv2.getTrackbarPos("sat", name);
        value = cv2.getTrackbarPos("value", name);
        #switch = cv2.getTrackbarPos("0 : OFF \n 1 : ON", name);
        
        lower = np.array([hue, sat, value]);
        upper = np.array([hue, 255, 255]);
        
        mask = cv2.inRange(hsv, lower, upper);
        
        cv2.imshow(name, kernel);
        
        # ESC for exit
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break;
        

cv2.destroyAllWindows();