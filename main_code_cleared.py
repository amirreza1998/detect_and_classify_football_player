from __future__ import print_function

import cv2
import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='output.mp4')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

##make substracktion
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))


if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)


##point creation
points1 = np.array([(164, 150),
                    (886, 150),
                    (525, 0),
                    (525, 700)]).astype(np.float32)

points2 = np.array([(149, 165),
                    (1140, 115),
                    (640, 110),
                    (875, 780)]).astype(np.float32)


## hemographic trasnform
H = cv2.getPerspectiveTransform(points2, points1)
print(H)
z=0
## initialtion output result video
I2 = cv2.imread('2D_field.png')
out = cv2.VideoWriter('eggs-reverse.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (I2.shape[1], I2.shape[0]))
# out1 = cv2.VideoWriter('eggs-reverse1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (I2.shape[1], I2.shape[0]))
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    # frame=cv2.GaussianBlur(frame, (3,3), 0);
    fgMask = backSub.apply(frame)
    ret, T = cv2.threshold(fgMask, 254, 255, cv2.THRESH_BINARY)#thsi code for destroy shadows

    ## erosion and dilation
    # erosion_kernel = np.ones((15,5), np.uint8);
    erosion_kernel=np.array([[0 ,0 ,1 ,0 ,0],
                            [0 ,0 ,1 ,1 ,0],
                            [0 ,1 ,1 ,1 ,0],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0 ,0 ,1 ,0 ,0]],np.uint8)
    dilat_kernel = np.array([[0, 0, 1, 0, 0],
                               [0, 0, 1, 1, 0],
                               [0, 1, 1, 1, 0],
                               [1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1],
                               [0, 1, 1, 1, 0],
                               [0, 0, 1, 1, 0],
                               [0, 0, 1, 0, 0]], np.uint8)
    kernel = np.ones((1,2),np.uint8)
    T1 = cv2.morphologyEx(T, cv2.MORPH_OPEN, dilat_kernel)
    erosioned = cv2.erode(T1, erosion_kernel);
    dilated=cv2.dilate(erosioned,dilat_kernel);

    n2,C2, stats2, centroids2=cv2.connectedComponentsWithStats(dilated)

    #put rules on connected component siza
    delet_array=list()
    n2new=n2
    #delet small connected componnent
    for i in range(0, n2 ):
        if stats2[i][4] < 150:
           delet_array.append(i)

           n2new=n2new-1
    centroids2new = np.delete(centroids2, delet_array,0)
    stats2=np.delete(stats2,delet_array, axis=0)

    ##delet connect componet that are not in yard and are uper than yard
    n2new1=n2new
    delet_array=list()
    for i in range(0, n2new - 1):
        if ((1225 - 20) / (60 - 120))*(centroids2new[i][1] - 60) + 1225 > centroids2new[i][0]:    # y meghdar avalie centroid ast va x meghdar dovom

            delet_array.append(i)
            n2new1=n2new1-1
    centroids2new = np.delete(centroids2new, delet_array,0)
    stats2 = np.delete(stats2, delet_array, axis=0)
    ##delet row and keep higth
    delet_array = list()
    for i in range(0, n2new1 - 1):
        if stats2[i][2]>stats2[i][3]:  # y meghdar avalie centroid ast va x meghdar dovom
            delet_array.append(i)
            n2new1 = n2new1 - 1
    centroids2new = np.delete(centroids2new, delet_array, 0)
    stats2 = np.delete(stats2, delet_array, axis=0)
    ## put text on frames
    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv.rectangle(fgMask, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(fgMask, str(n2new+1), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv.rectangle(dilated, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(dilated, str(n2new1+1), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    ##show images
    cv2.line(frame,(1225, 60),(20, 120),(255,0,255,1),2)
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    cv.imshow('thresholded',T)
    cv2.imshow('After Erosion', erosioned);
    cv2.imshow('After dilation', dilated);

    ## bird eye reading
    I1 = cv2.imread('2D_field.png')
    output_size=(I1.shape[0],I1.shape[1])
    eye_bird=I1
    i=1

    ##find location in 2d map and make player image cropped for learning part
    while i<n2new1:
        #transfer to 2d location of players
        location = np.array([((centroids2new[i][0]).astype(np.int32)),
                   ((centroids2new[i][1]+stats2[i][3]/2).astype(np.int32)),
                   (1)])
        location=location.transpose()
        location=np.dot(H,location)
        position1=location[0]/location[2]
        position2=location[1]/location[2]
        #crop player image
        cropped_image = frame[int(stats2[i][1]):int(stats2[i][1]+stats2[i][3]),int(stats2[i][0]):int(stats2[i][0]+stats2[i][2])]

        #make location of player on 2s map
        cv2.circle(eye_bird,(position1.astype(np.int32),position2.astype(np.int32)),10,(255,0,255,1),-1)
        cv2.circle(frame,((centroids2new[i][0].astype(np.int32)),(centroids2new[i][1].astype(np.int32))),10,(255,0,255,1),-1)
        i=i+1
    out.write(eye_bird)
    out1.write(dilated)
    cv2.imshow('eye bird',eye_bird)
    cv.imshow('Frame1', frame)
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
out.release()
out1.release()
