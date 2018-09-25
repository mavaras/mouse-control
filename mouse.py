import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0);     
while(1):
    try:
        ret, frame = cap.read();
        frame = cv2.flip(frame, 1);
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV);
        
        # define range of blue color in HSV
        lower = np.array([0,90,200]);
        upper = np.array([10,255,255]);
    
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower, upper);
    
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame, frame, mask = mask);
                
        # Find contours
        _,contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);
        cv2.drawContours(frame, contours, -1, (0,255,0), 2);
        cv2.circle(frame, (contours[0][0][0][0],50), 5, [0,0,255], -1);
        #cv2.circle(frame, (contours[0][0][0][0],contours[0][0][0][1]), 5, [0,0,255], -1);
        """
        if(contours != None):
            #cv2.drawContours(frame, contours, -1, (0,255,0), 3)
            miny = contours[0][0][0][0];
            aux = contours[0][0][0];
            for c in range(0, 10):
                if(contours[c][0][0][0] < miny):
                    miny = contours[c][0][0][0];
                    aux = contours[c][0][0];
            
            #print(aux)
            cv2.circle(frame, (aux[0],aux[1]), 5, [0,0,255], -1);
            #cv2.putText(frame, "Pon la mano", (0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, cv2.LINE_AA);
        
        #cv2.imshow("frame", frame);
        """
        """
        cnt = max(contours, key = lambda x: cv2.contourArea(x));
        epsilon = 0.0005 * cv2.arcLength(cnt, True);
        approx = cv2.approxPolyDP(cnt,epsilon, True);
        
        # Set convex hull around the hand
        hull = cv2.convexHull(cnt);
        
        # Define area of hull and area of hand
        hull_area = cv2.contourArea(hull);
        hand_area = cv2.contourArea(cnt);
      
        #find the percentage of area not covered by hand in convex hull
        arearatio = ((hull_area - hand_area) / hand_area)*100;
    
        # Get the defects in convex hull
        hull = cv2.convexHull(approx, returnPoints = False);
        defects = cv2.convexityDefects(approx, hull);
        
        for c in range(defects.shape[0]):
            s, e, f, d = defects[c, 0];
            start = tuple(approx[s][0]);
            end = tuple(approx[e][0]);
            far = tuple(approx[f][0]);
            
            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2);
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2);
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2);
            s = (a + b + c) / 2;
            ar = math.sqrt(s * (s - a) * (s - b) * (s - c));
            
            cv2.circle(frame, far, 5, [0,0,255], -1);
            cv2.circle(frame, (far[0], far[1]-150), 3, [255,0,0], -1);
        """
        cv2.imshow("frame", frame);
        cv2.imshow("res", res);

    except:
        pass;

    # ESC for exit
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
cv2.destroyAllWindows()
cap.release()        