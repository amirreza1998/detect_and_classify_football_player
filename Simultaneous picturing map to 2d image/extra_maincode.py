from __future__ import print_function

import cv2
import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='camera2.mkv')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

##make substracktion
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

capture1 = cv.VideoCapture(cv.samples.findFileOrKeep('camera1.mkv'))
capture0 = cv.VideoCapture(cv.samples.findFileOrKeep('camera0.mkv'))
capture2 = cv.VideoCapture(cv.samples.findFileOrKeep('camera2.mkv'))





#point creation  camera1
points1 = np.array([(164, 150),
                    (886, 150),
                    (525, 0),
                    (525, 700)]).astype(np.float32)

points2 = np.array([(149, 165),
                    (1140, 115),
                    (640, 110),
                    (875, 780)]).astype(np.float32)
## hemographic trasnform
H1 = cv2.getPerspectiveTransform(points2, points1)


##point creation  camera2
points1cam2 = np.array([(1050, 0),
                    (1050, 700),
                    (886, 553),
                    (886, 150)]).astype(np.float32)

points2cam2 = np.array([(511, 178),
                    (1252, 272),
                    (880, 280),
                    (457, 210)]).astype(np.float32)
## hemographic trasnform
H2 = cv2.getPerspectiveTransform(points2cam2, points1cam2)


##point creation
##point camera0
points1 = np.array([(164, 150),
                    (163, 553),
                    (0, 700),
                    (0, 0)]).astype(np.float32)

points2 = np.array([(897, 190),
                    (475, 295),
                    (50, 300),
                    (817, 160)]).astype(np.float32)
## hemographic trasnform
H0 = cv2.getPerspectiveTransform(points2, points1)
## initialtion output result video
I2 = cv2.imread('2D_field.png')
out = cv2.VideoWriter('eggs-reverse.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (I2.shape[1], I2.shape[0]))
out1 = cv2.VideoWriter('eggs-reverse1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (I2.shape[1], I2.shape[0]))
while True:
    ret, frame1 = capture1.read()
    ret, frame0 = capture0.read()
    ret, frame2 = capture2.read()


    # frame=cv2.GaussianBlur(frame, (3,3), 0);
    fgMask1 = backSub.apply(frame1)
    fgMask0 = backSub.apply(frame0)
    fgMask2 = backSub.apply(frame2)

    ret, T1 = cv2.threshold(fgMask1, 254, 255, cv2.THRESH_BINARY)#thsi code for destroy shadows
    ret, T0 = cv2.threshold(fgMask0, 254, 255, cv2.THRESH_BINARY)#thsi code for destroy shadows
    ret, T2 = cv2.threshold(fgMask2, 254, 255, cv2.THRESH_BINARY)#thsi code for destroy shadows

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
    erosioned1 = cv2.erode(T1, erosion_kernel);
    erosioned0 = cv2.erode(T0, erosion_kernel);
    erosioned2 = cv2.erode(T2, erosion_kernel);

    dilated1=cv2.dilate(erosioned1,dilat_kernel);
    dilated0=cv2.dilate(erosioned0,dilat_kernel);
    dilated2=cv2.dilate(erosioned2,dilat_kernel);

    n21,C21, stats21, centroids21=cv2.connectedComponentsWithStats(dilated1)
    n20,C20, stats20, centroids20=cv2.connectedComponentsWithStats(dilated0)
    n22,C22, stats22, centroids22=cv2.connectedComponentsWithStats(dilated2)

    #put rules on connected component siza
    delet_array=list()
    n2new1=n21
    #delet small connected componnent
    for i in range(0, n21 ):
        if stats21[i][4] < 150:
           delet_array.append(i)

           n2new1=n2new1-1
    centroids2new1 = np.delete(centroids21, delet_array,0)
    stats21=np.delete(stats21,delet_array, axis=0)
    delet_array=list()
    n2new2=n22
    #delet small connected componnent
    for i in range(0, n22 ):
        if stats22[i][4] < 150:
           delet_array.append(i)

           n2new2=n2new2-1
    centroids2new2 = np.delete(centroids22, delet_array,0)
    stats22=np.delete(stats22,delet_array, axis=0)
    delet_array=list()
    n2new0=n20
    #delet small connected componnent
    for i in range(0, n20 ):
        if stats20[i][4] < 150:
           delet_array.append(i)

           n2new0=n2new0-1
    centroids2new0 = np.delete(centroids20, delet_array,0)
    stats20=np.delete(stats20,delet_array, axis=0)


    ##show images
    cv.imshow('Frame1', frame1)
    cv.imshow('Frame0', frame0)
    cv.imshow('Frame2', frame2)
    cv2.imshow('After dilation 1', dilated1);
    cv2.imshow('After dilation 0', dilated0);
    cv2.imshow('After dilation 2', dilated2);

    ## bird eye reading
    I1 = cv2.imread('2D_field.png')
    output_size=(I1.shape[0],I1.shape[1])
    eye_bird=I1
    i=1

    ##find location in 2d map and make player image cropped for learning part
    while i<n2new1:
        #trans
        # er to 2d location of players
        location1 = np.array([(centroids2new1[i][0].astype(np.int32)),
                   (centroids2new1[i][1].astype(np.int32)),
                   (1)])
        location1=location1.transpose()
        location1=np.dot(H1,location1)
        position11=location1[0]/location1[2]
        position21=location1[1]/location1[2]
        #transfer to 2d location of players
        location2 = np.array([(centroids2new2[i][0].astype(np.int32)),
                   (centroids2new2[i][1].astype(np.int32)),
                   (1)])
        location2=location2.transpose()
        location2=np.dot(H2,location2)
        position12=location2[0]/location1[2]
        position22=location2[1]/location1[2]
        #transfer to 2d location of players
        location0 = np.array([(centroids2new0[i][0].astype(np.int32)),
                   (centroids2new0[i][1].astype(np.int32)),
                   (1)])
        location0=location0.transpose()
        location0=np.dot(H0,location0)
        position10=location0[0]/location0[2]
        position20=location0[1]/location0[2]

        #crop player image
        # cropped_image = frame[int(stats2[i][1]):int(stats2[i][1]+stats2[i][3]),int(stats2[i][0]):int(stats2[i][0]+stats2[i][2])]

        #make location of player on 2s map
        cv2.circle(eye_bird,(position11.astype(np.int32),position21.astype(np.int32)),10,(255,0,255,1),-1);
        cv2.circle(eye_bird,(position10.astype(np.int32),position20.astype(np.int32)),10,(255,0,255,1),-1);
        # cv2.circle(eye_bird,(position12.astype(np.int32),position22.astype(np.int32)),10,(255,0,255,1),-1);

        i=i+1
    # out.write(eye_bird)
    # out1.write(dilated)
    cv2.imshow('eye bird',eye_bird)
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
# out.release()
# out1.release()
