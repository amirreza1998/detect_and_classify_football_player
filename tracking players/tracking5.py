from __future__ import print_function

import cv2
import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
from random import randint

trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.legacy.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.legacy.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.legacy.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.legacy.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.legacy.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker

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

# Create a video capture object to read videos
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
success, frame = capture.read()
# quit if unable to read the video file
if not success:
    print('Failed to read video')
    sys.exit(1)

    ## Select boxes
bboxes = []
colors = []

# OpenCV's selectROI function doesn't work for selecting multiple objects in Python
# So we will call this function in a loop till we are done selecting all objects
while True:
    # draw bounding boxes over objects
    # selectROI's default behaviour is to draw box starting from the center
    # when fromCenter is set to false, you can draw box starting from top left corner
    bbox = cv2.selectROI('MultiTracker', frame)
    bboxes.append(bbox)
    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    print("Press q to quit selecting boxes and start tracking")
    print("Press any other key to select next object")
    k = cv2.waitKey(0) & 0xFF
    print(k)
    if (k == 113):  # q is pressed
        break
print(bbox)
# bbox=[(770, 244, 27, 56), (667, 147, 21, 41), (647, 138, 17, 34), (780, 138, 16, 39), (967, 134, 17, 41), (913, 101, 20, 35), (792, 102, 16, 23), (673, 105, 15, 29), (718, 100, 10, 27), (725, 104, 11, 25), (708, 97, 9, 20), (605, 97, 9, 24), (518, 100, 15, 24), (510, 127, 12, 34), (461, 100, 11, 17), (446, 111, 15, 33), (434, 129, 12, 36), (442, 184, 18, 44), (395, 180, 19, 42)]
print('Selected bounding boxes {}'.format(bboxes))
trackerType = "CSRT"
createTrackerByName(trackerType)

# Create MultiTracker object
multiTracker = cv2.legacy.MultiTracker_create()

# Initialize MultiTracker
for bbox in bboxes:
    multiTracker.add(createTrackerByName(trackerType), frame, bbox)

while capture.isOpened():
    ret, frame = capture.read()
    if frame is None:
        break
    timer = cv2.getTickCount()
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    ##other part
    # frame=cv2.GaussianBlur(frame, (3,3), 0);
    fgMask = backSub.apply(frame)
    ##tracking
    ret, boxes = multiTracker.update(frame)

    # draw tracked objects
    for i, newbox in enumerate(boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

    # show frame
    # Display tracker type on frame


    cv2.imshow('MultiTracker', frame)

    # quit on ESC button
    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break
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
    I1 = cv2.imread('../2D_field.png')
    output_size=(I1.shape[0],I1.shape[1])
    eye_bird=I1
    i=1

    ##find location in 2d map and make player image cropped for learning part
    while i<n2new1:
        #transfer to 2d location of players
        location = np.array([(centroids2new[i][0].astype(np.int32)),
                   (centroids2new[i][1].astype(np.int32)),
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
    cv2.imshow('eye bird',eye_bird)
    cv.imshow('Frame1', frame)
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

