import cv2
import numpy as np
from ultralytics import YOLO
#import video_filter #Script para mejorar la calidad del video

cap = cv2.VideoCapture('/Users/carloshernandez/Desktop/verano/pedestrian_dataset_1.mp4') #Video input
model = YOLO('yolov8m.pt') #Calling the YOLO CNN
obj_center = [] #Creo un arreglo vacio para guardar las posiciones del pedestrian

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

    for cls, bbox in zip(classes,bboxes): #cls is for classes

        (x,y,x2,y2) =bbox #Posiciones de las bounding boxes, y si pongo 4 xs y Y
        cv2.rectangle(frame,(x,y),(x2,y2),(0,0,255),2) #RGB: 0G, 0B, 255Red 2x2Pix
        cv2.putText(frame, str(cls), (x,y-5), cv2.FONT_HERSHEY_COMPLEX ,1,(0,0,255) ,2) #Write the class that it belongs to
        #Valore de las coordenadas en la consola
        #print("x: ",x, " y: ",y)
        #print("x2: ",x2, " y2: ",y2)
        
        #Punto medio del rectangulo
        x_center = int((x+x2)/2)
        y_center = int((y+y2)/2)
        #Dibujo un circulo en la mitad del rectangulo para ver la trayectoria
        cv2.circle(frame, ((x + x2) // 2, (y + y2) // 2), 4, (0, 255, 0), -1) 
        #Escribo la posición X y Y de los peatones
        cv2.putText(frame,('x:'+str(x_center)+'y:'+(str(y_center))),(x+14,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0))


  
        
    #print(bboxes)
    cv2.imshow('dataset',frame)
    #key = cv2.waitKey(1)
    #if key == 27:
    if cv2.waitKey(1) & 0xFF == ord('q'): #waitKey is 1ms between frames
        break

    
cap.release()
cv2.destroyAllWindows()



