import cv2
import numpy as np

# Create our body classifier
# features of pre_trained haarcascade full body
body_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

# Initiate video capture for video file
cap = cv2.VideoCapture('F:/AI assignment/AI assignment/Computer Vision and Image Processing_module_8/walking.avi')

# Get width and height of the frame of video
width  = cap.get(3) # float width
height = cap.get(4) # float height 

# Make a video writer to see if video being taken as input inflict any changes you make
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out_video = cv2.VideoWriter('result/output.avi', fourcc, 20.0, (int(width), int(height)), True)

# Loop once video is successfully loaded
while cap.isOpened():
    
    # Read first frame
    ret, frame = cap.read()
    #frame = cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow('Pedestrians', frame)

    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
 
cap.release()
cv2.destroyAllWindows()
