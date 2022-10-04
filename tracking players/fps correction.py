from __future__ import print_function

import cv2
import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt

capture = cv.VideoCapture('../result of neural learning.avi')
I2 = cv2.imread('../2D_field.png')
w=I2.shape[1]
h=I2.shape[0]
out = cv2.VideoWriter('fast_result.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (w, h))
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    timer = cv2.getTickCount()
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    cv2.putText(frame, 'svm' + " clasification", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (250, 170, 50), 2);

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (250, 170, 50), 2);
    timer = cv2.getTickCount()
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    out.write(frame)
    # cv2.imshow(frame)
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

out.release()
