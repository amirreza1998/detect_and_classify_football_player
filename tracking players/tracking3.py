import cv2
import numpy as np

# image_path
video = cv2.VideoCapture("output.mp4")
ok, frame = video.read()

# read image

# select ROIs function
ROIs = cv2.selectROIs("Select Rois", frame)

# print rectangle points of selected roi
print(ROIs)

# Crop selected roi ffrom raw image

# counter to save image with different name
crop_number = 0

# loop over every bounding box save in array "ROIs"
for rect in ROIs:
    x1 = rect[0]
    y1 = rect[1]
    x2 = rect[2]
    y2 = rect[3]

    # crop roi from original image
    img_crop = frame[y1:y1 + y2, x1:x1 + x2]

    # show cropped image
    cv2.imshow("crop" + str(crop_number), img_crop)

    # save cropped image
    cv2.imwrite("crop" + str(crop_number) + ".jpeg", img_crop)

    crop_number += 1

# hold window
cv2.waitKey(0)