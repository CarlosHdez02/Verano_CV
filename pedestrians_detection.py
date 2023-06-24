import cv2
import numpy as np
#import matplotlib.pyplot as plt
#import ultralytics as ul
from ultralytics import YOLO

cap = cv2.VideoCapture('pedestrian_dataset_1.mp4') #Video input

model = YOLO('yolov8m.pt') #Calling the YOLO CNN

while True:
    ret, frame = cap.read() #loopin the video
    if not ret:
        break

    results =model(frame)
    #print(result)
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype= "int") #bounding_boxes encapsules the object, that show me the x & y position of each box
    #Now, that we have encapsuled the object, what if i show the class that it belongs to
    classes = np.array(result.boxes.cls.cpu(), dtype="int")

    for cls, bbox in zip(classes,bboxes):

        (x,y,x2,y2) =bbox
        cv2.rectangle(frame,(x,y),(x2,y2),(0,0,255),2) #RGB: 0G, 0B, 255Red 2x2Pix
        cv2.putText(frame, str(cls), (x,y-5), cv2.FONT_HERSHEY_COMPLEX ,1,(0,0,255) ,2) #Write the class that it belongs to
        #print("x: ",x, " y: ",y)
        #print("x2: ",x2, " y2: ",y2)
    
    #print(bboxes)
    cv2.imshow('dataset',frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()