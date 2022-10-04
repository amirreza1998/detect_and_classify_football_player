import cv2
import sys

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if __name__ == '__main__':
    desired_tracker = 'CSRT'

    # Set up tracker.
    # Instead of MIL, you can also use

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[7]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()

    # Read video
    multiTracker = cv2.legacy.MultiTracker_create()
    video = cv2.VideoCapture("output.mp4")
    ok, frame = video.read()
    bounding_box_list = []
    # if ok:
    #
    #     while True:
    #
    #         # Draw a bounding box over all the objects that you want to track_type
    #         # Press ENTER or SPACE after you've drawn the bounding box
    #         bounding_box = cv2.selectROI('Multi-Object Tracker', frame,)
    #
    #         # Add a bounding box
    #         bounding_box_list.append(bounding_box)
    #         # Press 'q' (make sure you click on the video frame so that it is the
    #         # active window) to start object tracking. You can press another key
    #         # if you want to draw another bounding box.
    #         print("\nPress q to begin tracking objects or press " +
    #               "another key to draw the next bounding box\n")
    #
    #         # Wait for keypress
    #         k = cv2.waitKey() & 0xFF
    #
    #         # Start tracking objects if 'q' is pressed
    #         if k == ord('q'):
    #             break
    #
    #     cv2.destroyAllWindows()
    #     for bbox in bounding_box_list:
    #         print(bbox)
    #         # Add tracker to the multi-object tracker
    #         multiTracker.add( frame, bbox)
            # multiTracker.add()
    # Process the video
    # multiTracker = cv2.legacy.MultiTracker_create()

    bbox = cv2.selectROIs('hi',frame, False)
    print('hi',bbox)
    # Initialize tracker with first frame and bounding box
    for rect in bbox:
        print('bye',rect)
        multiTracker.add(cv2.TrackerCSRT_create(), frame, bbox)
        # multiTracker.add()

        ok = tracker.init(frame, rect)
        print(ok)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        for bbox in bbox:
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]
        # Update tracker
            ok, bbox = tracker.update(frame)


        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break