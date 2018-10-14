# CALIBRATION MODULE

import cv2
import numpy as np
import math
import pyautogui as pya
from pynput.mouse import Button, Controller

from skimage.filters.rank import maximum, minimum, median
from skimage.morphology import square
from skimage.exposure import equalize_hist
from skimage.color import rgb2gray


# functions definition
def nothing(x):
    pass;

# Calculates the intersection point between two given lines
def line_intersection(line1, line2):
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0]);
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]);

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0];

    x = 0;
    y = 0;
    div = det(x_diff, y_diff);
    if div != 0:
        d = (det(*line1), det(*line2));
        x = det(d, x_diff) / div;
        y = det(d, y_diff) / div;

    return x, y;

# Defines mask based on lower and upper color range
# Operations with the mask
def create_mask(lower_range, upper_range):
    mask = cv2.inRange(hsl, lower, upper);

    # Morphology: Open + Close
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8));
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8));

    mask = cv2.erode(mask, np.ones((5, 5)), iterations = 2);
    #mask = maximum(mask, square(5));
    #mask = minimum(mask, square(3));

    return mask;

# Shows windows
def display_windows():
    cv2.imshow("calibration", mask);
    cv2.imshow("frame", frame);

# Creates and set the calibration GUI
def create_calibration_gui(n):
    cv2.namedWindow(n);
    cv2.resizeWindow(n, 300, 300);
    cv2.createTrackbar("hue", n, 0, 255, nothing);
    cv2.createTrackbar("sat", n, 0, 255, nothing);
    cv2.createTrackbar("lightness", n, 0, 255, nothing);


# main block
if __name__ == "__main__":
    window_name = "calibration";
    mouse = Controller();
    camera_dim = [640, 480];
    region_of_interest = [580, 420];
    start = False;
    mouse_pos_prev = (0, 0);
    flag = False;

    create_calibration_gui(window_name);

    # Getting video capture
    cap = cv2.VideoCapture(1);
    cap.set(cv2.CAP_PROP_BRIGHTNESS, -2);
    print("frame brightness:",cv2.CAP_PROP_BRIGHTNESS);

    # Main loop
    while(1):
        ret, frame = cap.read();
        frame = np.array(frame, dtype = np.uint8);
        frame = cv2.flip(frame, 1);

        cv2.rectangle(frame, (camera_dim[0] - region_of_interest[0], camera_dim[1] - region_of_interest[1]),
                             (region_of_interest[0], region_of_interest[1]),
                      (0, 255, 0), 2);

        # Converting from RGB (default) to HSL color space
        hsl = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS);

        # Get trackbars HSL vaules, hue, saturation, lightness (tonalidad, saturación, brillo)
        hue = cv2.getTrackbarPos("hue", window_name);
        sat = cv2.getTrackbarPos("sat", window_name);
        lig = cv2.getTrackbarPos("lightness", window_name);

        # red -> 111 58 40

        # Setting color filter
        lower = np.array([hue, sat, lig]);
        upper = np.array([180, 255, 255]);

        mask = create_mask(lower, upper);
        #frame = hsl
        # Working with the contours of the mask
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2);

        points = [];
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt);
            #print("cnt len (nº of points)",len(cnt));

            cnt_area = cv2.contourArea(cnt);
            print("cnt_area:",cnt_area)
            diagonal_d = math.hypot(x - x + w, y - y + h); # diagonal length
            if cnt_area > 650:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2);
                cv2.line(frame, (x, y), (x + w, y + h), (255, 0, 0), 2);
                cv2.line(frame, (x + w, y), (x, y + h), (255, 0, 0), 2);
                cx, cy = line_intersection(((x, y), (x + w, y + h)), ((x + w, y), (x, y + h)));
                points.append((int(cx), int(cy)));
                cv2.circle(frame, (int(cx), int(cy)), 4, [32, 255, 255], -1);

            if len(points) >= 2:
                print("_______________________________________")
                flag = True;
                print("->",points[0][0],".",points[0][1])
                print("->", points[1][0], ",", points[1][1])
                print("->",(int(points[0][0] + points[1][0] / 2),
                            int(points[0][1] + points[1][1] / 2)))
                aux = points[0][0] + points[1][0] / 2;
                cv2.line(frame, points[0], points[1], (0, 255, 255), 2);
                point = (int((points[0][0] + points[1][0]) / 2),
                         int((points[0][1] + points[1][1]) / 2));
                cv2.circle(frame, point, 4, [32, 0, 255], -1);
                points.clear();

                s = "("+str(point[0])+","+str(point[1])+")";
                cv2.putText(frame, s, (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA);
                if start:
                    #pya.moveTo(4*(int(cx)-130), 4*(int(cy)-130), 0.00001, pya.easeInOutQuad);
                    mouse_pos = (4 * (int(cx) - 170), 4 * (int(cy) - 170));
                    mouse_pos = (4 * (point[0] - 170), 4 * (point[1] - 170));
                    print("mouse position:",mouse_pos);
                    if math.hypot(mouse_pos[0] - mouse_pos_prev[0], mouse_pos[1] - mouse_pos_prev[1]) > 10:
                        mouse.position = mouse_pos;
                        mouse_pos_prev = mouse_pos;

                cv2.putText(frame, "N", (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA);

            elif len(points) == 1 and cnt_area > 2000:
                print("____________mira!");
                cv2.putText(frame, "S", (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA);
                flag = False;
                points.clear();

        # Showing the lower colour range chosen in the trackbars
        cv2.circle(frame, (50, 50), 35, [hue, sat, lig], -1);

        # Displaying windows
        display_windows();

        # ESC for exit, s for start mouse control
        k = cv2.waitKey(5) & 0xFF;
        if k == 27:
            break;
        elif k == ord("s"):
            if start:
                start = False;
            else:
                start = True;
        elif k == ord("v"):
            mouse.click(Button.left, 1);

    # Releasing video capture
    cap.release();

    # Closing windows
    cv2.destroyAllWindows();
