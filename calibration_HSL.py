# CALIBRATION MODULE

import cv2
import numpy as np
from skimage.filters.rank import maximum, minimum, median
from skimage.morphology import square
from skimage.exposure import equalize_hist
from skimage.color import rgb2gray


def nothing(x):
    pass;

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0]);
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]);

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception("Error in line_intersection");

    d = (det(*line1), det(*line2));
    x = det(d, xdiff) / div;
    y = det(d, ydiff) / div;

    return x, y;

kernel = np.ones((5,5),np.uint8);

# Creating calibration GUI
name = "calibration";
cv2.namedWindow(name);
cv2.resizeWindow(name, 400, 400);
cv2.createTrackbar("hue", name, 0, 255, nothing);
cv2.createTrackbar("sat", name, 0, 255, nothing);
cv2.createTrackbar("lightness", name, 0, 255, nothing);

# Getting video capture
cap = cv2.VideoCapture(0);

# Main loop
while(1):
    ret, frame = cap.read();
    frame = np.array(frame, dtype = np.uint8);
    frame = cv2.flip(frame, 1);

    # Converting from RGB (default) to HSL color space
    hsl = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS);

    # Get trackbars HSL vaules, hue, saturation, lightness (tonalidad, saturación, brillo)
    hue = cv2.getTrackbarPos("hue", name);
    sat = cv2.getTrackbarPos("sat", name);
    lig = cv2.getTrackbarPos("lightness", name);

    # Setting color filter
    lower = np.array([hue, sat, lig]);
    upper = np.array([255,255,255]);
    
    mask = cv2.inRange(hsl, lower, upper);
    #mask = cv2.equalizeHist(mask);

    # Morphology: Opening + Close
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8));
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8));

    # Working with contours of the mask
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2);

    for c in contours:
        x, y, w, h = cv2.boundingRect(c);
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2);
        cv2.line(frame, (x, y), (x+w, y+h), (255, 0, 0), 2);
        cv2.line(frame, (x + w, y), (x, y + h), (255, 0, 0), 2);
        cx, cy = line_intersection(((x, x + w), (y, y + h)), ((x+w, x), (y, y + h)));
        cv2.circle(frame, (int(cx), int(cy)), 6, [0, 0, 255], -1);

    # Showing the lower colour range chosen in the trackbars
    cv2.circle(frame, (50,50), 35, [hue-20, sat, lig], -1);

    # Showing windows
    cv2.imshow(name, mask);
    #cv2.imshow("hsl", hsl);
    cv2.imshow("frame", frame);
    
    # ESC for exit
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break;

# Releasing video capture
cap.release();

# Closing windows
cv2.destroyAllWindows();