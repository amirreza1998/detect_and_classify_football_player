import cv2
import numpy as np
#find point of 2D_field.png
I1 = cv2.imread('../2D_field.png')
output_size = (I1.shape[0], I1.shape[1])
eye_bird = I1
cv2.circle(eye_bird, (164, 150), 0, (255, 0, 255, 1), 10, 0)# noghte goshe mohvate samt chap bala #this#cam0
cv2.circle(eye_bird, (886, 150), 0, (255, 0, 255, 1), 10, 0)# noghte goshe mohvate samt rast bala#this
cv2.circle(eye_bird, (0, 0), 0, (255, 0, 255, 1), 10, 0)# noghte goshe zamin samt chap bala
cv2.circle(eye_bird, (1050, 0), 0, (255, 0, 255, 1), 10, 0)#noghte goshe zamin samt rast bala
cv2.circle(eye_bird, (525, 0), 0, (255, 0, 255, 1), 10, 0)#noghte vasat zamin bala #this
cv2.circle(eye_bird, (525, 700), 0, (255, 0, 255, 1), 10, 0)#noghte vasat zamin pain #this
cv2.circle(eye_bird, (886, 553), 0, (255, 0, 255, 1), 10, 0)#noghte goshe mohavate rast payin
cv2.circle(eye_bird, (163, 553), 0, (255, 0, 255, 1), 10, 0)#noghte goshe mohavate chap payin
cv2.circle(eye_bird, (0, 700), 0, (255, 0, 255, 1), 10, 0)# noghte goshe zamin samt chap payin
cv2.circle(eye_bird, (1050, 700), 0, (255, 0, 255, 1), 10, 0)#noghte goshe zamin samt rast payin


#find point of output.mp4
# capture = cv2.VideoCapture(cv2.samples.findFileOrKeep('output.mp4'));
# ret, frame = capture.read()
# output_size2=(frame.shape[0],frame.shape[1])
# cv2.circle(frame, (149, 165), 0, (255, 0, 255, 1), 10, 0)# noghte goshe mohvate samt chap #this
# cv2.circle(frame, (1140, 115), 0, (255, 0, 255, 1), 10, 0)# noghte goshe mohvate samt rast #this
# cv2.circle(frame, (20, 150), 0, (255, 0, 255, 1), 10, 0)# noghte goshe zamin samt chap
# cv2.circle(frame, (1225, 88), 0, (255, 0, 255, 1), 10, 0)#noghte goshe zamin samt rast
# cv2.circle(frame, (640, 110), 0, (255, 0, 255, 1), 10, 0)#noghte vasat zamin bala #this
# cv2.circle(frame, (875, 780), 0, (255, 0, 255, 1), 10, 0)#noghte vasat zamin pain #this

# m=(1225-20)/(88-150)
# #y meghdar avalie centroid ast va x meghdar dovom
# y=m(x-88)+1225
# ##point creation
#
# points1 = np.array([(164, 150),
#                     (886, 150),
#                     (525, 0),
#                     (525, 700)]).astype(np.float32)
#
# points2 = np.array([(149, 165),
#                     (1140, 115),
#                     (640, 110),
#                     (875, 780)]).astype(np.float32)
# ## hemographic trasnform
# H = cv2.getPerspectiveTransform(points1, points2)
# print(H)


#find point of camera0
# capture = cv2.VideoCapture(cv2.samples.findFileOrKeep('camera_movie/camera0.mkv'));
# ret, frame = capture.read()
# cv2.imshow('hi',frame)
# output_size2=(frame.shape[0],frame.shape[1])
# cv2.circle(frame, (817, 160), 0, (255, 0, 255, 1), 5, 0) #noghte goshe zamin samt chap bala
# cv2.circle(frame, (50, 300), 0, (255, 0, 255, 1), 5, 0) #noghte goshe zamin samt chap payin
# cv2.circle(frame, (475, 295), 0, (255, 0, 255, 1), 5, 0) #noghte goshe mohavate chap payin
# cv2.circle(frame, (897, 190), 0, (255, 0, 255, 1), 5, 0) #noghte goshe mohavate chap bala
# cv2.circle(frame, (640, 110), 0, (255, 0, 255, 1), 10, 0)#noghte vasat zamin bala #this
# cv2.circle(frame, (875, 780), 0, (255, 0, 255, 1), 10, 0)#noghte vasat zamin pain #this


# ##point creation
#
# points1 = np.array([(164, 150),
#                     (163, 553),
#                     (0, 700),
#                     (0, 0)]).astype(np.float32)
#
# points2 = np.array([(897, 190),
#                     (475, 295),
#                     (50, 300),
#                     (817, 160)]).astype(np.float32)
# ## hemographic trasnform
# H0 = cv2.getPerspectiveTransform(points1, points2)
# print(0)


#find point of camera2
capture = cv2.VideoCapture(cv2.samples.findFileOrKeep('camera_movie/camera2.mkv'));
ret, frame = capture.read()
# cv2.imshow('hi',frame)
output_size2=(frame.shape[0],frame.shape[1])
# cv2.circle(frame, (511, 178), 0, (255, 0, 255, 1), 5, 0) #noghte goshe zamin samt rast bala
# cv2.circle(frame, (1252, 272), 0, (255, 0, 255, 1), 5, 0) #noghte goshe zamin samt rast payin
# cv2.circle(frame, (880, 280), 0, (255, 0, 255, 1), 5, 0) #noghte goshe mohavate rast payin
# cv2.circle(frame, (457, 210), 0, (255, 0, 255, 1), 5, 0) #noghte goshe mohavate rast bala
# cv2.circle(frame, (640, 110), 0, (255, 0, 255, 1), 10, 0)#noghte vasat zamin bala #this
# cv2.circle(frame, (875, 780), 0, (255, 0, 255, 1), 10, 0)#noghte vasat zamin pain #this


##point creation

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
print(0)

cv2.imshow('capture',frame)
cv2.imshow('eye bird', eye_bird)
print(output_size)
print(output_size2)
cv2.waitKey(30000)
points1 = np.array([(164, 150),
                    (886, 150),
                    (525, 0),
                    (525, 700)]).astype(np.float32)
print(points1.transpose())
a = np.array([0,10,20,30,40, 50, 60, 70, 80, 90, 100])

print(a[2])